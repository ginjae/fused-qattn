#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "quantization_utils.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel for Block-wise Dequantization
// Dequantizes int8 weights to float using block-wise scale and zero-point
// W_quantized: [rows * cols] int8 values
// scales: [num_blocks] scale factors
// zero_points: [num_blocks] zero points
// W_dequantized: [rows * cols] output float values
// block_size: number of elements per quantization block
static __global__ void dequantize_blockwise_kernel(
    const int8_t* W_quantized,
    const float* scales,
    const int8_t* zero_points,
    float* W_dequantized,
    int rows,
    int cols,
    int block_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        // Determine which block this element belongs to
        int block_idx = idx / block_size;

        // Dequantize: float_value = scale * (quantized_value - zero_point)
        float scale = scales[block_idx];
        int8_t zero_point = zero_points[block_idx];
        int8_t quantized_val = W_quantized[idx];

        W_dequantized[idx] = scale * (float)(quantized_val - zero_point);
    }
}

// Kernel 0: Linear Projection (X·W + b)
// X: [batch, seq_len, d_model]
// W: [d_model, d_out]
// b: [d_out]
// Output: [batch, seq_len, d_out]
static __global__ void linear_projection_kernel(
    const float* X,
    const float* W,
    const float* b,
    float* output,
    int batch,
    int seq_len,
    int d_model,
    int d_out
) {
    int b_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_idx < batch && row < seq_len && col < d_out) {
        float sum = 0.0f;

        // Compute dot product of X[b_idx, row, :] and W[:, col]
        for (int k = 0; k < d_model; k++) {
            int x_idx = b_idx * seq_len * d_model + row * d_model + k;
            int w_idx = k * d_out + col;
            sum += X[x_idx] * W[w_idx];
        }

        // Add bias
        if (b != nullptr) {
            sum += b[col];
        }

        int out_idx = b_idx * seq_len * d_out + row * d_out + col;
        output[out_idx] = sum;
    }
}

// Kernel 1: Q·Kᵀ MatMul
// Q: [batch, seq_len, d_k]
// K: [batch, seq_len, d_k]
// Output: [batch, seq_len, seq_len]
static __global__ void qk_matmul_kernel(
    const float* Q,
    const float* K,
    float* QK,
    int batch,
    int seq_len,
    int d_k
) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && row < seq_len && col < seq_len) {
        float sum = 0.0f;

        // Compute dot product of Q[b, row, :] and K[b, col, :]
        for (int k = 0; k < d_k; k++) {
            int q_idx = b * seq_len * d_k + row * d_k + k;
            int k_idx = b * seq_len * d_k + col * d_k + k;
            sum += Q[q_idx] * K[k_idx];
        }

        int out_idx = b * seq_len * seq_len + row * seq_len + col;
        QK[out_idx] = sum;
    }
}

// Kernel 2: Scaling + Masking
// QK: [batch, seq_len, seq_len]
// scale_factor: 1/sqrt(d_k)
// causal_mask: true for causal attention
static __global__ void scale_mask_kernel(
    float* QK,
    int batch,
    int seq_len,
    float scale_factor,
    bool causal_mask
) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && row < seq_len && col < seq_len) {
        int idx = b * seq_len * seq_len + row * seq_len + col;

        // Scale
        QK[idx] *= scale_factor;

        // Apply causal mask (mask future positions)
        if (causal_mask && col > row) {
            QK[idx] = -INFINITY;
        }
    }
}

// Kernel 3: Softmax (row-wise)
// QK: [batch, seq_len, seq_len]
// Output: [batch, seq_len, seq_len]
static __global__ void softmax_kernel(
    const float* QK,
    float* A,
    int batch,
    int seq_len
) {
    int b = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && row < seq_len) {
        int base_idx = b * seq_len * seq_len + row * seq_len;

        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < seq_len; i++) {
            max_val = fmaxf(max_val, QK[base_idx + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float exp_val = expf(QK[base_idx + i] - max_val);
            A[base_idx + i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for (int i = 0; i < seq_len; i++) {
            A[base_idx + i] /= sum;
        }
    }
}

// Kernel 4: Attention Score Application (A·V)
// A: [batch, seq_len, seq_len] - attention scores
// V: [batch, seq_len, d_v]
// Output: [batch, seq_len, d_v]
static __global__ void av_matmul_kernel(
    const float* A,
    const float* V,
    float* output,
    int batch,
    int seq_len,
    int d_v
) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && row < seq_len && col < d_v) {
        float sum = 0.0f;

        // Compute dot product of A[b, row, :] and V[b, :, col]
        for (int k = 0; k < seq_len; k++) {
            int a_idx = b * seq_len * seq_len + row * seq_len + k;
            int v_idx = b * seq_len * d_v + k * d_v + col;
            sum += A[a_idx] * V[v_idx];
        }

        int out_idx = b * seq_len * d_v + row * d_v + col;
        output[out_idx] = sum;
    }
}

// Host function to perform complete naive attention with quantized weight matrices
void naive_attention_quantized(
    const float* d_X,
    const int8_t* d_Wq_quantized,
    const int8_t* d_Wk_quantized,
    const int8_t* d_Wv_quantized,
    const float* d_Wq_scales,
    const float* d_Wk_scales,
    const float* d_Wv_scales,
    const int8_t* d_Wq_zeros,
    const int8_t* d_Wk_zeros,
    const int8_t* d_Wv_zeros,
    const float* d_bq,
    const float* d_bk,
    const float* d_bv,
    float* d_output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    int block_size,
    bool causal_mask = false
) {
    // Allocate temporary buffers for dequantized weights
    float *d_Wq, *d_Wk, *d_Wv;
    CUDA_CHECK(cudaMalloc(&d_Wq, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv, d_model * d_v * sizeof(float)));

    // Dequantize weights
    dim3 block_dequant(16, 16);

    // Dequantize Wq
    dim3 grid_wq((d_k + block_dequant.x - 1) / block_dequant.x,
                 (d_model + block_dequant.y - 1) / block_dequant.y);
    dequantize_blockwise_kernel<<<grid_wq, block_dequant>>>(
        d_Wq_quantized, d_Wq_scales, d_Wq_zeros, d_Wq, d_model, d_k, block_size);
    CUDA_CHECK(cudaGetLastError());

    // Dequantize Wk
    dim3 grid_wk((d_k + block_dequant.x - 1) / block_dequant.x,
                 (d_model + block_dequant.y - 1) / block_dequant.y);
    dequantize_blockwise_kernel<<<grid_wk, block_dequant>>>(
        d_Wk_quantized, d_Wk_scales, d_Wk_zeros, d_Wk, d_model, d_k, block_size);
    CUDA_CHECK(cudaGetLastError());

    // Dequantize Wv
    dim3 grid_wv((d_v + block_dequant.x - 1) / block_dequant.x,
                 (d_model + block_dequant.y - 1) / block_dequant.y);
    dequantize_blockwise_kernel<<<grid_wv, block_dequant>>>(
        d_Wv_quantized, d_Wv_scales, d_Wv_zeros, d_Wv, d_model, d_v, block_size);
    CUDA_CHECK(cudaGetLastError());

    // Allocate temporary buffers for Q, K, V, QK, A
    float *d_Q, *d_K, *d_V, *d_QK, *d_A;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_QK, batch * seq_len * seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A, batch * seq_len * seq_len * sizeof(float)));

    dim3 block_proj(16, 16);

    // Step 0-1: Compute Q = X·Wq + bq (using dequantized weights)
    dim3 grid_q((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-2: Compute K = X·Wk + bk (using dequantized weights)
    dim3 grid_k((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-3: Compute V = X·Wv + bv (using dequantized weights)
    dim3 grid_v((d_v + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Step 1: Q·Kᵀ MatMul
    dim3 block1(16, 16);
    dim3 grid1((seq_len + block1.x - 1) / block1.x,
               (seq_len + block1.y - 1) / block1.y,
               batch);
    qk_matmul_kernel<<<grid1, block1>>>(d_Q, d_K, d_QK, batch, seq_len, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Scaling + Masking
    float scale_factor = 1.0f / sqrtf((float)d_k);
    scale_mask_kernel<<<grid1, block1>>>(d_QK, batch, seq_len, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Step 3: Softmax
    dim3 block2(256);
    dim3 grid2((seq_len + block2.x - 1) / block2.x, batch);
    softmax_kernel<<<grid2, block2>>>(d_QK, d_A, batch, seq_len);
    CUDA_CHECK(cudaGetLastError());

    // Step 4: A·V MatMul
    dim3 block3(16, 16);
    dim3 grid3((d_v + block3.x - 1) / block3.x,
               (seq_len + block3.y - 1) / block3.y,
               batch);
    av_matmul_kernel<<<grid3, block3>>>(d_A, d_V, d_output, batch, seq_len, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_QK));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Original host function (kept for backward compatibility)
void naive_attention(
    const float* d_X,
    const float* d_Wq,
    const float* d_Wk,
    const float* d_Wv,
    const float* d_bq,
    const float* d_bk,
    const float* d_bv,
    float* d_output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    bool causal_mask = false
) {
    // Allocate temporary buffers for Q, K, V, QK, A
    float *d_Q, *d_K, *d_V, *d_QK, *d_A;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_QK, batch * seq_len * seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A, batch * seq_len * seq_len * sizeof(float)));

    dim3 block_proj(16, 16);

    // Step 0-1: Compute Q = X·Wq + bq
    dim3 grid_q((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-2: Compute K = X·Wk + bk
    dim3 grid_k((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-3: Compute V = X·Wv + bv
    dim3 grid_v((d_v + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Step 1: Q·Kᵀ MatMul
    dim3 block1(16, 16);
    dim3 grid1((seq_len + block1.x - 1) / block1.x,
               (seq_len + block1.y - 1) / block1.y,
               batch);
    qk_matmul_kernel<<<grid1, block1>>>(d_Q, d_K, d_QK, batch, seq_len, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Scaling + Masking
    float scale_factor = 1.0f / sqrtf((float)d_k);
    scale_mask_kernel<<<grid1, block1>>>(d_QK, batch, seq_len, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Step 3: Softmax
    dim3 block2(256);
    dim3 grid2((seq_len + block2.x - 1) / block2.x, batch);
    softmax_kernel<<<grid2, block2>>>(d_QK, d_A, batch, seq_len);
    CUDA_CHECK(cudaGetLastError());

    // Step 4: A·V MatMul
    dim3 block3(16, 16);
    dim3 grid3((d_v + block3.x - 1) / block3.x,
               (seq_len + block3.y - 1) / block3.y,
               batch);
    av_matmul_kernel<<<grid3, block3>>>(d_A, d_V, d_output, batch, seq_len, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_QK));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaDeviceSynchronize());
}

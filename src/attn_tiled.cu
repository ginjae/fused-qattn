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
__global__ void dequantize_blockwise_kernel(
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

// Kernel 0: Tiled Linear Projection (X·W + b)
// X: [batch, seq_len, d_model]
// W: [d_model, d_out]
// b: [d_out]
// Output: [batch, seq_len, d_out]
#define TILE_SIZE 16

__global__ void linear_projection_tiled_kernel(
    const float* X,
    const float* W,
    const float* b,
    float* output,
    int batch,
    int seq_len,
    int d_model,
    int d_out
) {
    __shared__ float X_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float W_tile[TILE_SIZE][TILE_SIZE];

    int b_idx = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (d_model + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load X tile
        int x_row = row;
        int x_col = t * TILE_SIZE + threadIdx.x;
        if (b_idx < batch && x_row < seq_len && x_col < d_model) {
            X_tile[threadIdx.y][threadIdx.x] = X[b_idx * seq_len * d_model + x_row * d_model + x_col];
        } else {
            X_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load W tile
        int w_row = t * TILE_SIZE + threadIdx.y;
        int w_col = col;
        if (w_row < d_model && w_col < d_out) {
            W_tile[threadIdx.y][threadIdx.x] = W[w_row * d_out + w_col];
        } else {
            W_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += X_tile[threadIdx.y][k] * W_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (b_idx < batch && row < seq_len && col < d_out) {
        if (b != nullptr) {
            sum += b[col];
        }
        output[b_idx * seq_len * d_out + row * d_out + col] = sum;
    }
}

// Kernel 1: Tiled Q·K^T MatMul
// Q: [batch, seq_len, d_k]
// K: [batch, seq_len, d_k]
// Output: [batch, seq_len, seq_len]
__global__ void qk_matmul_tiled_kernel(
    const float* Q,
    const float* K,
    float* QK,
    int batch,
    int seq_len,
    int d_k
) {
    __shared__ float Q_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE];

    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (d_k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load Q tile
        int q_row = row;
        int q_col = t * TILE_SIZE + threadIdx.x;
        if (b < batch && q_row < seq_len && q_col < d_k) {
            Q_tile[threadIdx.y][threadIdx.x] = Q[b * seq_len * d_k + q_row * d_k + q_col];
        } else {
            Q_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load K tile (transposed)
        int k_row = col;
        int k_col = t * TILE_SIZE + threadIdx.y;
        if (b < batch && k_row < seq_len && k_col < d_k) {
            K_tile[threadIdx.y][threadIdx.x] = K[b * seq_len * d_k + k_row * d_k + k_col];
        } else {
            K_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Q_tile[threadIdx.y][k] * K_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (b < batch && row < seq_len && col < seq_len) {
        QK[b * seq_len * seq_len + row * seq_len + col] = sum;
    }
}

// Kernel 2: Scaling + Masking
// QK: [batch, seq_len, seq_len]
// scale_factor: 1/sqrt(d_k)
// causal_mask: true for causal attention
__global__ void scale_mask_kernel(
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

// Kernel 3: Chunk-wise Softmax
// QK: [batch, seq_len, seq_len]
// Output: [batch, seq_len, seq_len]
// Uses shared memory for chunk-based computation to improve memory access patterns
#define CHUNK_SIZE 256

__global__ void softmax_kernel(
    const float* QK,
    float* A,
    int batch,
    int seq_len
) {
    __shared__ float shared_max[CHUNK_SIZE];
    __shared__ float shared_sum[CHUNK_SIZE];

    int b = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (b >= batch || row >= seq_len) return;

    int base_idx = b * seq_len * seq_len + row * seq_len;
    int num_chunks = (seq_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Phase 1: Find global max across all chunks
    float local_max = -INFINITY;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int idx = chunk * CHUNK_SIZE + tid;
        if (idx < seq_len) {
            local_max = fmaxf(local_max, QK[base_idx + idx]);
        }
    }
    shared_max[tid] = local_max;
    __syncthreads();

    // Reduce to find global max
    for (int s = CHUNK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    float global_max = shared_max[0];
    __syncthreads();

    // Phase 2: Compute exp and partial sums for each chunk
    float local_sum = 0.0f;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int idx = chunk * CHUNK_SIZE + tid;
        if (idx < seq_len) {
            float exp_val = expf(QK[base_idx + idx] - global_max);
            A[base_idx + idx] = exp_val;
            local_sum += exp_val;
        }
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduce to find global sum
    for (int s = CHUNK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float global_sum = shared_sum[0];
    __syncthreads();

    // Phase 3: Normalize each chunk
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int idx = chunk * CHUNK_SIZE + tid;
        if (idx < seq_len) {
            A[base_idx + idx] /= global_sum;
        }
    }
}

// Kernel 4: Tiled Attention Score Application (A·V)
// A: [batch, seq_len, seq_len] - attention scores
// V: [batch, seq_len, d_v]
// Output: [batch, seq_len, d_v]
__global__ void av_matmul_tiled_kernel(
    const float* A,
    const float* V,
    float* output,
    int batch,
    int seq_len,
    int d_v
) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float V_tile[TILE_SIZE][TILE_SIZE];

    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (seq_len + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (b < batch && a_row < seq_len && a_col < seq_len) {
            A_tile[threadIdx.y][threadIdx.x] = A[b * seq_len * seq_len + a_row * seq_len + a_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load V tile
        int v_row = t * TILE_SIZE + threadIdx.y;
        int v_col = col;
        if (b < batch && v_row < seq_len && v_col < d_v) {
            V_tile[threadIdx.y][threadIdx.x] = V[b * seq_len * d_v + v_row * d_v + v_col];
        } else {
            V_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[threadIdx.y][k] * V_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (b < batch && row < seq_len && col < d_v) {
        output[b * seq_len * d_v + row * d_v + col] = sum;
    }
}

// Host function to perform complete tiled attention with quantized weight matrices
void tiled_attention_quantized(
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

    dim3 block_proj(TILE_SIZE, TILE_SIZE);

    // Step 0-1: Compute Q = X·Wq + bq (using dequantized weights with tiled matmul)
    dim3 grid_q((d_k + TILE_SIZE - 1) / TILE_SIZE,
                (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                batch);
    linear_projection_tiled_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-2: Compute K = X·Wk + bk (using dequantized weights with tiled matmul)
    dim3 grid_k((d_k + TILE_SIZE - 1) / TILE_SIZE,
                (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                batch);
    linear_projection_tiled_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-3: Compute V = X·Wv + bv (using dequantized weights with tiled matmul)
    dim3 grid_v((d_v + TILE_SIZE - 1) / TILE_SIZE,
                (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                batch);
    linear_projection_tiled_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Step 1: Q·Kᵀ MatMul with tiling
    dim3 block1(TILE_SIZE, TILE_SIZE);
    dim3 grid1((seq_len + TILE_SIZE - 1) / TILE_SIZE,
               (seq_len + TILE_SIZE - 1) / TILE_SIZE,
               batch);
    qk_matmul_tiled_kernel<<<grid1, block1>>>(d_Q, d_K, d_QK, batch, seq_len, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Scaling + Masking
    float scale_factor = 1.0f / sqrtf((float)d_k);
    scale_mask_kernel<<<grid1, block1>>>(d_QK, batch, seq_len, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Step 3: Chunk-wise Softmax
    dim3 block2(CHUNK_SIZE);
    dim3 grid2(seq_len, batch);
    softmax_kernel<<<grid2, block2>>>(d_QK, d_A, batch, seq_len);
    CUDA_CHECK(cudaGetLastError());

    // Step 4: A·V MatMul
    dim3 block3(16, 16);
    dim3 grid3((d_v + block3.x - 1) / block3.x,
               (seq_len + block3.y - 1) / block3.y,
               batch);
    av_matmul_tiled_kernel<<<grid3, block3>>>(d_A, d_V, d_output, batch, seq_len, d_v);
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
void tiled_attention(
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

    dim3 block_proj(TILE_SIZE, TILE_SIZE);

    // Step 0-1: Compute Q = X·Wq + bq with tiled matmul
    dim3 grid_q((d_k + TILE_SIZE - 1) / TILE_SIZE,
                (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                batch);
    linear_projection_tiled_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-2: Compute K = X·Wk + bk with tiled matmul
    dim3 grid_k((d_k + TILE_SIZE - 1) / TILE_SIZE,
                (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                batch);
    linear_projection_tiled_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-3: Compute V = X·Wv + bv with tiled matmul
    dim3 grid_v((d_v + TILE_SIZE - 1) / TILE_SIZE,
                (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                batch);
    linear_projection_tiled_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Step 1: Q·Kᵀ MatMul with tiling
    dim3 block1(TILE_SIZE, TILE_SIZE);
    dim3 grid1((seq_len + TILE_SIZE - 1) / TILE_SIZE,
               (seq_len + TILE_SIZE - 1) / TILE_SIZE,
               batch);
    qk_matmul_tiled_kernel<<<grid1, block1>>>(d_Q, d_K, d_QK, batch, seq_len, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Scaling + Masking
    float scale_factor = 1.0f / sqrtf((float)d_k);
    scale_mask_kernel<<<grid1, block1>>>(d_QK, batch, seq_len, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Step 3: Chunk-wise Softmax
    dim3 block2(CHUNK_SIZE);
    dim3 grid2(seq_len, batch);
    softmax_kernel<<<grid2, block2>>>(d_QK, d_A, batch, seq_len);
    CUDA_CHECK(cudaGetLastError());

    // Step 4: A·V MatMul
    dim3 block3(16, 16);
    dim3 grid3((d_v + block3.x - 1) / block3.x,
               (seq_len + block3.y - 1) / block3.y,
               batch);
    av_matmul_tiled_kernel<<<grid3, block3>>>(d_A, d_V, d_output, batch, seq_len, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_QK));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaDeviceSynchronize());
}


// Simple test function
int main() {
    // Test parameters (GPT-2 small scale)
    int batch = 1;
    int seq_len = 128;
    int d_model = 768;
    int d_k = 64;
    int d_v = 64;
    int block_size = 64; // Block size for quantization

    printf("=== Starting Test: Tiled Attention with Quantization (GPT-2 Scale) ===");

    // Allocate and initialize host memory for input and weights
    size_t x_size = batch * seq_len * d_model * sizeof(float);
    size_t wq_size = d_model * d_k * sizeof(float);
    size_t wk_size = d_model * d_k * sizeof(float);
    size_t wv_size = d_model * d_v * sizeof(float);
    size_t bq_size = d_k * sizeof(float);
    size_t bk_size = d_k * sizeof(float);
    size_t bv_size = d_v * sizeof(float);
    size_t out_size = batch * seq_len * d_v * sizeof(float);

    float* h_X = (float*)malloc(x_size);
    float* h_Wq = (float*)malloc(wq_size);
    float* h_Wk = (float*)malloc(wk_size);
    float* h_Wv = (float*)malloc(wv_size);
    float* h_bq = (float*)malloc(bq_size);
    float* h_bk = (float*)malloc(bk_size);
    float* h_bv = (float*)malloc(bv_size);
    float* h_output = (float*)malloc(out_size);

    srand(42);  // For reproducibility

    // Initialize input X
    for (int i = 0; i < batch * seq_len * d_model; i++) {
        h_X[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Range: [-1, 1]
    }

    // Initialize weight matrices with deterministic random values
    for (int i = 0; i < d_model * d_k; i++) {
        h_Wq[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;    // Range: [-0.05, 0.05]
        h_Wk[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
    for (int i = 0; i < d_model * d_v; i++) {
        h_Wv[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }

    // Initialize biases with deterministic random values
    for (int i = 0; i < d_k; i++) {
        h_bq[i] = ((float)rand() / RAND_MAX) * 0.01f - 0.005f;  // Range: [-0.005, 0.005]
        h_bk[i] = ((float)rand() / RAND_MAX) * 0.01f - 0.005f;
    }
    for (int i = 0; i < d_v; i++) {
        h_bv[i] = ((float)rand() / RAND_MAX) * 0.01f - 0.005f;
    }

    // Allocate device memory
    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_bq, *d_bk, *d_bv, *d_output;
    CUDA_CHECK(cudaMalloc(&d_X, x_size));
    CUDA_CHECK(cudaMalloc(&d_Wq, wq_size));
    CUDA_CHECK(cudaMalloc(&d_Wk, wk_size));
    CUDA_CHECK(cudaMalloc(&d_Wv, wv_size));
    CUDA_CHECK(cudaMalloc(&d_bq, bq_size));
    CUDA_CHECK(cudaMalloc(&d_bk, bk_size));
    CUDA_CHECK(cudaMalloc(&d_bv, bv_size));
    CUDA_CHECK(cudaMalloc(&d_output, out_size));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, x_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wq, h_Wq, wq_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk, h_Wk, wk_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv, h_Wv, wv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bq, h_bq, bq_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bk, h_bk, bk_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bv, h_bv, bv_size, cudaMemcpyHostToDevice));

    // Allocate output buffers for both tests
    float* h_output_unquant = (float*)malloc(out_size);
    float* h_output_quant = (float*)malloc(out_size);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed_time_unquant, elapsed_time_quant;

    // Test 1: Tiled attention with original (unquantized) weights
    printf("\n=== Test 1: Tiled Attention (Unquantized Weights) ===\n");

    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    tiled_attention(d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv, 
                    d_output, batch, seq_len, d_model, d_k, d_v, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    tiled_attention(d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv, 
                    d_output, batch, seq_len, d_model, d_k, d_v, false);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_unquant, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output_unquant, d_output, out_size, cudaMemcpyDeviceToHost));

    // Print results (first 3 and last 3 positions only)
    printf("Output (Unquantized) - showing first 3 and last 3 positions:\n");
    for (int i = 0; i < seq_len; i++) {
        if (i < 3 || i >= seq_len - 3) {
            printf("Position %d: [", i);
            for (int j = 0; j < (d_v < 8 ? d_v : 8); j++) {
                int idx = i * d_v + j;
                printf("%.6f", h_output_unquant[idx]);
                if (j < (d_v < 8 ? d_v : 8) - 1) printf(", ");
            }
            if (d_v > 8) printf(", ...");
            printf("]\n");
        } else if (i == 3) {
            printf("... (%d more positions) ...\n", seq_len - 6);
        }
    }
    printf("Execution time: %.4f ms\n", elapsed_time_unquant);

    // Quantize weights
    printf("\n=== Quantizing Weights ===\n");
    int num_blocks_q = (d_model * d_k + block_size - 1) / block_size;
    int num_blocks_v = (d_model * d_v + block_size - 1) / block_size;

    int8_t* h_Wq_quant = (int8_t*)malloc(d_model * d_k * sizeof(int8_t));
    int8_t* h_Wk_quant = (int8_t*)malloc(d_model * d_k * sizeof(int8_t));
    int8_t* h_Wv_quant = (int8_t*)malloc(d_model * d_v * sizeof(int8_t));

    float* h_Wq_scales = (float*)malloc(num_blocks_q * sizeof(float));
    float* h_Wk_scales = (float*)malloc(num_blocks_q * sizeof(float));
    float* h_Wv_scales = (float*)malloc(num_blocks_v * sizeof(float));

    int8_t* h_Wq_zeros = (int8_t*)malloc(num_blocks_q * sizeof(int8_t));
    int8_t* h_Wk_zeros = (int8_t*)malloc(num_blocks_q * sizeof(int8_t));
    int8_t* h_Wv_zeros = (int8_t*)malloc(num_blocks_v * sizeof(int8_t));

    quantize_blockwise(h_Wq, h_Wq_quant, h_Wq_scales, h_Wq_zeros, d_model * d_k, block_size);
    quantize_blockwise(h_Wk, h_Wk_quant, h_Wk_scales, h_Wk_zeros, d_model * d_k, block_size);
    quantize_blockwise(h_Wv, h_Wv_quant, h_Wv_scales, h_Wv_zeros, d_model * d_v, block_size);

    printf("Block size: %d\n", block_size);
    printf("Num blocks (Q/K): %d, Num blocks (V): %d\n", num_blocks_q, num_blocks_v);
    printf("Quantization complete.\n");

    // Test 2: Tiled attention with quantized weights (GPU dequantization)
    printf("\n=== Test 2: Tiled Attention (Quantized Weights) ===\n");

    // Allocate device memory for quantized weights
    int8_t *d_Wq_quant, *d_Wk_quant, *d_Wv_quant;
    float *d_Wq_scales, *d_Wk_scales, *d_Wv_scales;
    int8_t *d_Wq_zeros, *d_Wk_zeros, *d_Wv_zeros;

    CUDA_CHECK(cudaMalloc(&d_Wq_quant, d_model * d_k * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Wk_quant, d_model * d_k * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Wv_quant, d_model * d_v * sizeof(int8_t)));

    CUDA_CHECK(cudaMalloc(&d_Wq_scales, num_blocks_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk_scales, num_blocks_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv_scales, num_blocks_v * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Wq_zeros, num_blocks_q * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Wk_zeros, num_blocks_q * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Wv_zeros, num_blocks_v * sizeof(int8_t)));

    // Copy quantized data to device
    CUDA_CHECK(cudaMemcpy(d_Wq_quant, h_Wq_quant, d_model * d_k * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk_quant, h_Wk_quant, d_model * d_k * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv_quant, h_Wv_quant, d_model * d_v * sizeof(int8_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_Wq_scales, h_Wq_scales, num_blocks_q * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk_scales, h_Wk_scales, num_blocks_q * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv_scales, h_Wv_scales, num_blocks_v * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_Wq_zeros, h_Wq_zeros, num_blocks_q * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk_zeros, h_Wk_zeros, num_blocks_q * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv_zeros, h_Wv_zeros, num_blocks_v * sizeof(int8_t), cudaMemcpyHostToDevice));

    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    tiled_attention_quantized(d_X, 
                              d_Wq_quant, d_Wk_quant, d_Wv_quant,
                              d_Wq_scales, d_Wk_scales, d_Wv_scales,
                              d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              block_size, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    // Run quantized attention
    tiled_attention_quantized(d_X, 
                              d_Wq_quant, d_Wk_quant, d_Wv_quant,
                              d_Wq_scales, d_Wk_scales, d_Wv_scales,
                              d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              block_size, false);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_quant, start, stop));

    // Copy quantized result back
    CUDA_CHECK(cudaMemcpy(h_output_quant, d_output, out_size, cudaMemcpyDeviceToHost));

    // Print quantized results (first 3 and last 3 positions only)
    printf("Output (Quantized) - showing first 3 and last 3 positions:\n");
    for (int i = 0; i < seq_len; i++) {
        if (i < 3 || i >= seq_len - 3) {
            printf("Position %d: [", i);
            for (int j = 0; j < (d_v < 8 ? d_v : 8); j++) {
                int idx = i * d_v + j;
                printf("%.6f", h_output_quant[idx]);
                if (j < (d_v < 8 ? d_v : 8) - 1) printf(", ");
            }
            if (d_v > 8) printf(", ...");
            printf("]\n");
        } else if (i == 3) {
            printf("... (%d more positions) ...\n", seq_len - 6);
        }
    }
    printf("Execution time: %.4f ms\n", elapsed_time_quant);

    // Compare results
    printf("\n=== Comparison: Unquantized vs Quantized ===\n");
    float max_diff = 0.0f;
    float sum_sq_diff = 0.0f;
    int total_elements = batch * seq_len * d_v;

    for (int i = 0; i < total_elements; i++) {
        float diff = fabsf(h_output_unquant[i] - h_output_quant[i]);
        max_diff = fmaxf(max_diff, diff);
        sum_sq_diff += diff * diff;
    }

    float rmse = sqrtf(sum_sq_diff / total_elements);

    printf("Max absolute difference: %.8f\n", max_diff);
    printf("RMSE: %.8f\n", rmse);
    printf("\n=== Performance Comparison ===\n");
    printf("Unquantized execution time: %.4f ms\n", elapsed_time_unquant);
    printf("Quantized execution time:   %.4f ms\n", elapsed_time_quant);
    printf("\n");

    // Cleanup CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Cleanup
    free(h_X);
    free(h_Wq);
    free(h_Wk);
    free(h_Wv);
    free(h_bq);
    free(h_bk);
    free(h_bv);
    free(h_output);
    free(h_output_unquant);
    free(h_output_quant);
    free(h_Wq_quant);
    free(h_Wk_quant);
    free(h_Wv_quant);
    free(h_Wq_scales);
    free(h_Wk_scales);
    free(h_Wv_scales);
    free(h_Wq_zeros);
    free(h_Wk_zeros);
    free(h_Wv_zeros);

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaFree(d_bq));
    CUDA_CHECK(cudaFree(d_bk));
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_Wq_quant));
    CUDA_CHECK(cudaFree(d_Wk_quant));
    CUDA_CHECK(cudaFree(d_Wv_quant));
    CUDA_CHECK(cudaFree(d_Wq_scales));
    CUDA_CHECK(cudaFree(d_Wk_scales));
    CUDA_CHECK(cudaFree(d_Wv_scales));
    CUDA_CHECK(cudaFree(d_Wq_zeros));
    CUDA_CHECK(cudaFree(d_Wk_zeros));
    CUDA_CHECK(cudaFree(d_Wv_zeros));

    printf("=== End of Test ===\n");

    return 0;
}

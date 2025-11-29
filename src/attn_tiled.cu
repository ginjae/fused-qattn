#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "quantization_utils.cuh"

// Kernel 1: Tiled Q·K^T MatMul
// Q: [batch, seq_len, d_k]
// K: [batch, seq_len, d_k]
// Output: [batch, seq_len, seq_len]
static __global__ void qk_matmul_tiled_kernel(
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

// Kernel 3: Chunk-wise Softmax
// QK: [batch, seq_len, seq_len]
// Output: [batch, seq_len, seq_len]
// Uses shared memory for chunk-based computation to improve memory access patterns
#define CHUNK_SIZE 256

static __global__ void softmax_kernel(
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
static __global__ void av_matmul_tiled_kernel(
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


// Host function to perform complete tiled attention with quantized weight matrices
void tiled_attention_quantized_blockwise(
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

// Host function to perform naive attention with MXFP4 quantized weight matrices
void tiled_attention_mxfp4(
    const float* d_X,
    const void* d_Wq_mxfp4,
    const void* d_Wk_mxfp4,
    const void* d_Wv_mxfp4,
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
    const MXFP4Block* Wq_blocks = static_cast<const MXFP4Block*>(d_Wq_mxfp4);
    const MXFP4Block* Wk_blocks = static_cast<const MXFP4Block*>(d_Wk_mxfp4);
    const MXFP4Block* Wv_blocks = static_cast<const MXFP4Block*>(d_Wv_mxfp4);

    // Allocate temporary buffers for dequantized weights
    float *d_Wq, *d_Wk, *d_Wv;
    CUDA_CHECK(cudaMalloc(&d_Wq, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv, d_model * d_v * sizeof(float)));

    // Dequantize MXFP4 weights
    int total_size_q = d_model * d_k;
    int total_size_v = d_model * d_v;
    
    dim3 block_dequant(256);
    dim3 grid_wq((total_size_q + block_dequant.x - 1) / block_dequant.x);
    dim3 grid_wk((total_size_q + block_dequant.x - 1) / block_dequant.x);
    dim3 grid_wv((total_size_v + block_dequant.x - 1) / block_dequant.x);

    dequantize_mxfp4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq, total_size_q);
    CUDA_CHECK(cudaGetLastError());

    dequantize_mxfp4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk, total_size_q);
    CUDA_CHECK(cudaGetLastError());

    dequantize_mxfp4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv, total_size_v);
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
    linear_projection_tiled_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-2: Compute K = X·Wk + bk (using dequantized weights)
    dim3 grid_k((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_tiled_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-3: Compute V = X·Wv + bv (using dequantized weights)
    dim3 grid_v((d_v + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_tiled_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Step 1: Q·Kᵀ MatMul
    dim3 block1(16, 16);
    dim3 grid1((seq_len + block1.x - 1) / block1.x,
               (seq_len + block1.y - 1) / block1.y,
               batch);
    qk_matmul_tiled_kernel<<<grid1, block1>>>(d_Q, d_K, d_QK, batch, seq_len, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Scaling + Masking
    float scale_factor = 1.0f / sqrtf((float)d_k);
    scale_mask_kernel<<<grid1, block1>>>(d_QK, batch, seq_len, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Step 3: Softmax
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

// Host function to perform naive attention with NF4 quantized weight matrices
void tiled_attention_nf4(
    const float* d_X,
    const void* d_Wq_nf4,
    const void* d_Wk_nf4,
    const void* d_Wv_nf4,
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
    const NF4Block* Wq_blocks = static_cast<const NF4Block*>(d_Wq_nf4);
    const NF4Block* Wk_blocks = static_cast<const NF4Block*>(d_Wk_nf4);
    const NF4Block* Wv_blocks = static_cast<const NF4Block*>(d_Wv_nf4);

    // Allocate temporary buffers for dequantized weights
    float *d_Wq, *d_Wk, *d_Wv;
    CUDA_CHECK(cudaMalloc(&d_Wq, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv, d_model * d_v * sizeof(float)));

    // Dequantize NF4 weights
    int total_size_q = d_model * d_k;
    int total_size_v = d_model * d_v;
    
    dim3 block_dequant(256);
    dim3 grid_wq((total_size_q + block_dequant.x - 1) / block_dequant.x);
    dim3 grid_wk((total_size_q + block_dequant.x - 1) / block_dequant.x);
    dim3 grid_wv((total_size_v + block_dequant.x - 1) / block_dequant.x);

    dequantize_nf4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq, total_size_q);
    CUDA_CHECK(cudaGetLastError());

    dequantize_nf4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk, total_size_q);
    CUDA_CHECK(cudaGetLastError());

    dequantize_nf4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv, total_size_v);
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
    linear_projection_tiled_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-2: Compute K = X·Wk + bk (using dequantized weights)
    dim3 grid_k((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_tiled_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-3: Compute V = X·Wv + bv (using dequantized weights)
    dim3 grid_v((d_v + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_tiled_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Step 1: Q·Kᵀ MatMul
    dim3 block1(16, 16);
    dim3 grid1((seq_len + block1.x - 1) / block1.x,
               (seq_len + block1.y - 1) / block1.y,
               batch);
    qk_matmul_tiled_kernel<<<grid1, block1>>>(d_Q, d_K, d_QK, batch, seq_len, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Scaling + Masking
    float scale_factor = 1.0f / sqrtf((float)d_k);
    scale_mask_kernel<<<grid1, block1>>>(d_QK, batch, seq_len, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Step 3: Softmax
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

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

// Kernel 0: Fused & Tiled Linear Projection for Q, K, V
// Computes [Q|K|V] = X·[W_Q|W_K|W_V] + [b_Q|b_K|b_V] in a single kernel
// This improves arithmetic intensity by loading X once for all three projections
// X: [batch, seq_len, d_model]
// Wq, Wk, Wv: [d_model, d_k/d_v]
// bq, bk, bv: [d_k/d_v]
// Q, K, V: [batch, seq_len, d_k/d_v]
#define TILE_SIZE 16

static __global__ void fused_qkv_projection_tiled_kernel(
    const float* X,
    const float* Wq,
    const float* Wk,
    const float* Wv,
    const float* bq,
    const float* bk,
    const float* bv,
    float* Q,
    float* K,
    float* V,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v
) {
    __shared__ float X_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float W_tile[TILE_SIZE][TILE_SIZE];

    int b_idx = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum_q = 0.0f;
    float sum_k = 0.0f;
    float sum_v = 0.0f;

    // Loop over tiles of the shared d_model dimension
    for (int t = 0; t < (d_model + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load X tile (shared for all three projections)
        int x_row = row;
        int x_col = t * TILE_SIZE + threadIdx.x;
        if (b_idx < batch && x_row < seq_len && x_col < d_model) {
            X_tile[threadIdx.y][threadIdx.x] = X[b_idx * seq_len * d_model + x_row * d_model + x_col];
        } else {
            X_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Compute Q projection
        int w_row = t * TILE_SIZE + threadIdx.y;
        int w_col = col;
        if (w_row < d_model && w_col < d_k) {
            W_tile[threadIdx.y][threadIdx.x] = Wq[w_row * d_k + w_col];
        } else {
            W_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum_q += X_tile[threadIdx.y][k] * W_tile[k][threadIdx.x];
        }
        __syncthreads();

        // Compute K projection (reuse X_tile)
        if (w_row < d_model && w_col < d_k) {
            W_tile[threadIdx.y][threadIdx.x] = Wk[w_row * d_k + w_col];
        } else {
            W_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum_k += X_tile[threadIdx.y][k] * W_tile[k][threadIdx.x];
        }
        __syncthreads();

        // Compute V projection (reuse X_tile)
        if (w_row < d_model && w_col < d_v) {
            W_tile[threadIdx.y][threadIdx.x] = Wv[w_row * d_v + w_col];
        } else {
            W_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum_v += X_tile[threadIdx.y][k] * W_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write Q result
    if (b_idx < batch && row < seq_len && col < d_k) {
        if (bq != nullptr) {
            sum_q += bq[col];
        }
        Q[b_idx * seq_len * d_k + row * d_k + col] = sum_q;
    }

    // Write K result
    if (b_idx < batch && row < seq_len && col < d_k) {
        if (bk != nullptr) {
            sum_k += bk[col];
        }
        K[b_idx * seq_len * d_k + row * d_k + col] = sum_k;
    }

    // Write V result
    if (b_idx < batch && row < seq_len && col < d_v) {
        if (bv != nullptr) {
            sum_v += bv[col];
        }
        V[b_idx * seq_len * d_v + row * d_v + col] = sum_v;
    }
}

#define WARP_SIZE 32
#define TILE_D 32  // Tile size in d_k dimension

// Q: [batch, seq_len, d_k]
// K: [batch, seq_len, d_k]
// V: [batch, seq_len, d_v]
// output: [batch, seq_len, d_v]
// grid: (seq_len, batch), blockDim.x = e.g., 256
// shared memory: seq_len * sizeof(float) bytes (logits buffer)
static __global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int batch,
    int seq_len,
    int d_k,
    int d_v,
    float scale_factor,
    bool causal_mask
) {
    // One block handles one (b, row) pair
    int b   = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (b >= batch || row >= seq_len) return;

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Static shared memory
    __shared__ float q_tile[TILE_D];      // Tile of Q[row] in d_k dimension
    __shared__ float shared_max;
    __shared__ float shared_sum;
    __shared__ float warp_maxes[32];      // Max 32 warps (assuming blockDim.x <= 1024)
    __shared__ float warp_sums[32];

    // Dynamic shared memory: logits[row, :]
    // Launch with shared_bytes = seq_len * sizeof(float)
    extern __shared__ float logits[];     // size: seq_len

    // Base indices
    int q_base   = b * seq_len * d_k + row * d_k;
    int k_base   = b * seq_len * d_k;
    int v_base   = b * seq_len * d_v;
    int out_base = b * seq_len * d_v + row * d_v;

    // Step 1: Initialize logits(row, :)
    for (int col = tid; col < seq_len; col += blockDim.x) {
        logits[col] = 0.0f;
    }
    __syncthreads();

    // Step 2: Compute QK^T (tiling along d_k axis, using shared q_tile)
    for (int t = 0; t < d_k; t += TILE_D) {
        // Load Q[row, t:t+TILE_D) tile into shared memory
        if (tid < TILE_D) {
            int dim = t + tid;
            if (dim < d_k) {
                q_tile[tid] = Q[q_base + dim];
            } else {
                q_tile[tid] = 0.0f;
            }
        }
        __syncthreads();

        // Accumulate partial dot products for columns handled by this thread
        for (int col = tid; col < seq_len; col += blockDim.x) {
            float acc = 0.0f;
            int k_offset = k_base + col * d_k + t;

            for (int k = 0; k < TILE_D; ++k) {
                int dim = t + k;
                if (dim < d_k) {
                    float q_val = q_tile[k];
                    float k_val = K[k_offset + k];
                    acc += q_val * k_val;
                }
            }

            logits[col] += acc;
        }
        __syncthreads();
    }

    // Step 3: Scaling + causal mask + compute row-wise max
    float thread_max = -INFINITY;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = logits[col] * scale_factor;

        if (causal_mask && col > row) {
            val = -INFINITY;
        }

        logits[col] = val;
        thread_max = fmaxf(thread_max, val);
    }

    // Warp-level max reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    // Store each warp's result in shared memory
    if (lane_id == 0) {
        warp_maxes[warp_id] = thread_max;
    }
    __syncthreads();

    // First warp reduces warp_maxes
    float block_max = -INFINITY;
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        block_max = warp_maxes[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
    }
    if (warp_id == 0 && lane_id == 0) {
        shared_max = block_max;
    }
    __syncthreads();

    float global_max = shared_max;

    // Step 4: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = logits[col];

        if (causal_mask && col > row) {
            logits[col] = 0.0f;
            continue;
        }

        float exp_val = expf(val - global_max);
        logits[col]   = exp_val;
        thread_sum   += exp_val;
    }

    // Warp-level sum reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        block_sum = warp_sums[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
    if (warp_id == 0 && lane_id == 0) {
        shared_sum = block_sum;
    }
    __syncthreads();

    float global_sum = shared_sum;

    // Step 5: Softmax normalization (logits = A[row, :])
    if (global_sum > 0.0f) {
        for (int col = tid; col < seq_len; col += blockDim.x) {
            logits[col] /= global_sum;
        }
    } else {
        for (int col = tid; col < seq_len; col += blockDim.x) {
            logits[col] = 0.0f;
        }
    }
    __syncthreads();

    // Step 6: A[row, :] · V => output[b, row, :]
    // Each thread handles part of the d_v dimension
    for (int dv = tid; dv < d_v; dv += blockDim.x) {
        float acc = 0.0f;
        for (int col = 0; col < seq_len; ++col) {
            float a = logits[col];  // A[row, col]
            float v = V[v_base + col * d_v + dv];
            acc += a * v;
        }
        output[out_base + dv] = acc;
    }
}


// Host function to perform complete tiled attention with quantized weight matrices
void flash_attention_quantized(
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

    // Allocate temporary buffers for Q, K, V, A (no need for QK anymore!)
    float *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));
        
    // Step 0: Compute Q, K, V using fused kernel for better arithmetic intensity
    // Grid dimensions based on max(d_k, d_v) to handle all outputs
    int max_d = (d_k > d_v) ? d_k : d_v;
    dim3 block_fused(TILE_SIZE, TILE_SIZE);
    dim3 grid_fused((max_d + TILE_SIZE - 1) / TILE_SIZE,
                    (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                    batch);
    fused_qkv_projection_tiled_kernel<<<grid_fused, block_fused>>>(
        d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv,
        d_Q, d_K, d_V,
        batch, seq_len, d_model, d_k, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Step 1 & 2 & 3 & 4 (FUSED): Q·Kᵀ MatMul + Scaling + Masking + Online Softmax + A·V MatMul
    dim3 block_fused_qk(256);  // One block per row with 256 threads
    dim3 grid_fused_qk(seq_len, batch);
    const float scale_factor = rsqrtf((float)d_k);

    size_t shared_bytes = seq_len * sizeof(float);
    flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
        d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Original host function (kept for backward compatibility)
void flash_attention(
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
    // Allocate temporary buffers for Q, K, V, A (no need for QK anymore!)
    float *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));

    // Step 0: Compute Q, K, V using fused kernel for better arithmetic intensity
    // Grid dimensions based on max(d_k, d_v) to handle all outputs
    int max_d = (d_k > d_v) ? d_k : d_v;
    dim3 block_fused(TILE_SIZE, TILE_SIZE);
    dim3 grid_fused((max_d + TILE_SIZE - 1) / TILE_SIZE,
                    (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                    batch);
    fused_qkv_projection_tiled_kernel<<<grid_fused, block_fused>>>(
        d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv,
        d_Q, d_K, d_V,
        batch, seq_len, d_model, d_k, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Step 1 & 2 & 3 & 4 (FUSED): Q·Kᵀ MatMul + Scaling + Masking + Online Softmax + A·V MatMul
    dim3 block_fused_qk(256);  // One block per row with 256 threads
    dim3 grid_fused_qk(seq_len, batch);
    const float scale_factor = rsqrtf((float)d_k);
    size_t shared_bytes = seq_len * sizeof(float);
    flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
        d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaDeviceSynchronize());
}

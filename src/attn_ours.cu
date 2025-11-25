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

// Kernel: Optimized Fused Dequantization + QKV Projection
// Performs on-the-fly dequantization during matrix multiplication
// Optimizations:
// - __restrict__ pointers for better memory aliasing hints
// - Padded shared memory (TILE_SIZE+1) to avoid bank conflicts
// - __ldg() intrinsics for read-only cached loads
// - __fmaf_rn() for faster fused multiply-add operations
// - const variables and pre-computed indices
// - Better loop unrolling (#pragma unroll 8)
// X: [batch, seq_len, d_model]
// Wq_quantized, Wk_quantized, Wv_quantized: [d_model, d_k/d_v] int8
// scales, zero_points: per-block quantization parameters
// Q, K, V: [batch, seq_len, d_k/d_v]
static __global__ void fused_dequant_qkv_projection_kernel(
    const float* __restrict__ X,
    const int8_t* __restrict__ Wq_quantized,
    const int8_t* __restrict__ Wk_quantized,
    const int8_t* __restrict__ Wv_quantized,
    const float* __restrict__ Wq_scales,
    const float* __restrict__ Wk_scales,
    const float* __restrict__ Wv_scales,
    const int8_t* __restrict__ Wq_zeros,
    const int8_t* __restrict__ Wk_zeros,
    const int8_t* __restrict__ Wv_zeros,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv,
    float* __restrict__ Q,
    float* __restrict__ K,
    float* __restrict__ V,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    int block_size
) {
    // Padded shared memory to avoid bank conflicts (stride is TILE_SIZE+1 instead of TILE_SIZE)
    __shared__ float X_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Wq_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Wk_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Wv_tile[TILE_SIZE][TILE_SIZE + 1];

    const int b_idx = blockIdx.z;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum_q = 0.0f;
    float sum_k = 0.0f;
    float sum_v = 0.0f;

    // Pre-compute base index for X access
    const int x_base = b_idx * seq_len * d_model + row * d_model;

    for (int t = 0; t < (d_model + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 1. Load X tile once
        const int x_col = t * TILE_SIZE + threadIdx.x;
        if (b_idx < batch && row < seq_len && x_col < d_model) {
            X_tile[threadIdx.y][threadIdx.x] = __ldg(&X[x_base + x_col]);
        } else {
            X_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        const int w_row = t * TILE_SIZE + threadIdx.y;
        const int w_col = col;

        // 2. Load ALL weight tiles in parallel (no sync between)
        if (w_row < d_model && w_col < d_k) {
            const int wq_idx = w_row * d_k + w_col;
            const int block_idx = wq_idx / block_size;
            const int8_t q_val = __ldg(&Wq_quantized[wq_idx]);
            const float scale = __ldg(&Wq_scales[block_idx]);
            const int8_t zero = __ldg(&Wq_zeros[block_idx]);
            Wq_tile[threadIdx.y][threadIdx.x] = scale * (float)(q_val - zero);

            // Load K at same time
            const int wk_idx = w_row * d_k + w_col;
            const int block_idx_k = wk_idx / block_size;
            const int8_t k_val = __ldg(&Wk_quantized[wk_idx]);
            const float scale_k = __ldg(&Wk_scales[block_idx_k]);
            const int8_t zero_k = __ldg(&Wk_zeros[block_idx_k]);
            Wk_tile[threadIdx.y][threadIdx.x] = scale_k * (float)(k_val - zero_k);
        } else {
            Wq_tile[threadIdx.y][threadIdx.x] = 0.0f;
            Wk_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (w_row < d_model && w_col < d_v) {
            const int wv_idx = w_row * d_v + w_col;
            const int block_idx_v = wv_idx / block_size;
            const int8_t v_val = __ldg(&Wv_quantized[wv_idx]);
            const float scale_v = __ldg(&Wv_scales[block_idx_v]);
            const int8_t zero_v = __ldg(&Wv_zeros[block_idx_v]);
            Wv_tile[threadIdx.y][threadIdx.x] = scale_v * (float)(v_val - zero_v);
        } else {
            Wv_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 3. Single sync after all loads
        __syncthreads();

        // 4. Compute all three matmuls together
        for (int k = 0; k < TILE_SIZE; k++) {
            float x_val = X_tile[threadIdx.y][k];
            sum_q = __fmaf_rn(x_val, Wq_tile[k][threadIdx.x], sum_q);
            sum_k = __fmaf_rn(x_val, Wk_tile[k][threadIdx.x], sum_k);
            sum_v = __fmaf_rn(x_val, Wv_tile[k][threadIdx.x], sum_v);
        }
        __syncthreads();
    }

    // Write Q result with bias (coalesced writes)
    if (b_idx < batch && row < seq_len && col < d_k) {
        if (bq != nullptr) {
            sum_q = __fmaf_rn(1.0f, __ldg(&bq[col]), sum_q);
        }
        Q[b_idx * seq_len * d_k + row * d_k + col] = sum_q;
    }

    // Write K result with bias
    if (b_idx < batch && row < seq_len && col < d_k) {
        if (bk != nullptr) {
            sum_k = __fmaf_rn(1.0f, __ldg(&bk[col]), sum_k);
        }
        K[b_idx * seq_len * d_k + row * d_k + col] = sum_k;
    }

    // Write V result with bias
    if (b_idx < batch && row < seq_len && col < d_v) {
        if (bv != nullptr) {
            sum_v = __fmaf_rn(1.0f, __ldg(&bv[col]), sum_v);
        }
        V[b_idx * seq_len * d_v + row * d_v + col] = sum_v;
    }
}

#define WARP_SIZE 32
#define TILE_D 32  // Tile size in d_k dimension
#define TILE_PAD 1  // Padding to avoid bank conflicts

// Q: [batch, seq_len, d_k]
// K: [batch, seq_len, d_k]
// V: [batch, seq_len, d_v]
// output: [batch, seq_len, d_v]
// grid: (seq_len, batch), blockDim.x = e.g., 256
// shared memory: seq_len * sizeof(float) bytes (logits buffer)
static __global__ void our_attention_kernel(
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
// Now uses fused dequantization + QKV projection kernel for maximum throughput
void our_attention_quantized(
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
    // Allocate temporary buffers for Q, K, V only (no need for dequantized weights!)
    float *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));
        
    // Step 0: Compute Q, K, V using FUSED dequantization + projection kernel
    // This performs on-the-fly dequantization during matmul, eliminating separate passes
    int max_d = (d_k > d_v) ? d_k : d_v;
    dim3 block_fused(TILE_SIZE, TILE_SIZE);
    dim3 grid_fused((max_d + TILE_SIZE - 1) / TILE_SIZE,
                    (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                    batch);
    fused_dequant_qkv_projection_kernel<<<grid_fused, block_fused>>>(
        d_X,
        d_Wq_quantized, d_Wk_quantized, d_Wv_quantized,
        d_Wq_scales, d_Wk_scales, d_Wv_scales,
        d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
        d_bq, d_bk, d_bv,
        d_Q, d_K, d_V,
        batch, seq_len, d_model, d_k, d_v, block_size);
    CUDA_CHECK(cudaGetLastError());

    // Step 1 & 2 & 3 & 4 (FUSED): Q·Kᵀ MatMul + Scaling + Masking + Online Softmax + A·V MatMul
    dim3 block_fused_qk(256);  // One block per row with 256 threads
    dim3 grid_fused_qk(seq_len, batch);
    const float scale_factor = rsqrtf((float)d_k);

    size_t shared_bytes = seq_len * sizeof(float);
    our_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
        d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Original host function (kept for backward compatibility)
void our_attention(
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
    our_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
        d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaDeviceSynchronize());
}

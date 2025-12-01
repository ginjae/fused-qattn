#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "quantization_utils.cuh"

#define WARP_SIZE 32
#define TILE_D 32  // Tile size in d_k dimension
#define TIMING_NUM_ITERATIONS 100

// Helper function to compute median for timing
static float compute_median_local(float* times, int n) {
    std::sort(times, times + n);
    if (n % 2 == 0) {
        return (times[n/2 - 1] + times[n/2]) / 2.0f;
    } else {
        return times[n/2];
    }
}

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

// Host function to perform complete tiled attention with quantized weight matrices
void flash_attention_quantized_blockwise(
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

void flash_attention_mxfp4(
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
    bool causal_mask = false,
    float* kernel_times = nullptr,
    int* num_kernels = nullptr
) {
    const MXFP4Block* Wq_blocks = static_cast<const MXFP4Block*>(d_Wq_mxfp4);
    const MXFP4Block* Wk_blocks = static_cast<const MXFP4Block*>(d_Wk_mxfp4);
    const MXFP4Block* Wv_blocks = static_cast<const MXFP4Block*>(d_Wv_mxfp4);

    const int NUM_KERNELS = 5;
    if (num_kernels) *num_kernels = NUM_KERNELS;

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
    
    // Allocate temporary buffers for Q, K, V, A (no need for QK anymore!)
    float *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));

    // Grid setup
    int max_d = (d_k > d_v) ? d_k : d_v;
    dim3 block_fused(TILE_SIZE, TILE_SIZE);
    dim3 grid_fused((max_d + TILE_SIZE - 1) / TILE_SIZE,
                    (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                    batch);
    dim3 block_fused_qk(256);
    dim3 grid_fused_qk(seq_len, batch);
    const float scale_factor = rsqrtf((float)d_k);
    size_t shared_bytes = seq_len * sizeof(float);

    if (kernel_times) {
        cudaEvent_t starts[NUM_KERNELS], stops[NUM_KERNELS];
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventCreate(&starts[i]));
            CUDA_CHECK(cudaEventCreate(&stops[i]));
        }
        
        float all_times[TIMING_NUM_ITERATIONS][NUM_KERNELS];
        
        for (int iter = 0; iter < TIMING_NUM_ITERATIONS; iter++) {
            CUDA_CHECK(cudaEventRecord(starts[0]));
            dequantize_mxfp4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq, total_size_q);
            CUDA_CHECK(cudaEventRecord(stops[0]));
            
            CUDA_CHECK(cudaEventRecord(starts[1]));
            dequantize_mxfp4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk, total_size_q);
            CUDA_CHECK(cudaEventRecord(stops[1]));
            
            CUDA_CHECK(cudaEventRecord(starts[2]));
            dequantize_mxfp4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv, total_size_v);
            CUDA_CHECK(cudaEventRecord(stops[2]));
            
            CUDA_CHECK(cudaEventRecord(starts[3]));
            fused_qkv_projection_tiled_kernel<<<grid_fused, block_fused>>>(
                d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv,
                d_Q, d_K, d_V,
                batch, seq_len, d_model, d_k, d_v);
            CUDA_CHECK(cudaEventRecord(stops[3]));
            
            CUDA_CHECK(cudaEventRecord(starts[4]));
            flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
                d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
            CUDA_CHECK(cudaEventRecord(stops[4]));
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            for (int k = 0; k < NUM_KERNELS; k++) {
                CUDA_CHECK(cudaEventElapsedTime(&all_times[iter][k], starts[k], stops[k]));
            }
        }
        
        for (int k = 0; k < NUM_KERNELS; k++) {
            float times_for_kernel[TIMING_NUM_ITERATIONS];
            for (int i = 0; i < TIMING_NUM_ITERATIONS; i++) {
                times_for_kernel[i] = all_times[i][k];
            }
            kernel_times[k] = compute_median_local(times_for_kernel, TIMING_NUM_ITERATIONS);
        }
        
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventDestroy(starts[i]));
            CUDA_CHECK(cudaEventDestroy(stops[i]));
        }
    } else {
        dequantize_mxfp4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq, total_size_q);
        CUDA_CHECK(cudaGetLastError());
        
        dequantize_mxfp4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk, total_size_q);
        CUDA_CHECK(cudaGetLastError());
        
        dequantize_mxfp4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv, total_size_v);
        CUDA_CHECK(cudaGetLastError());
        
        fused_qkv_projection_tiled_kernel<<<grid_fused, block_fused>>>(
            d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv,
            d_Q, d_K, d_V,
            batch, seq_len, d_model, d_k, d_v);
        CUDA_CHECK(cudaGetLastError());

        flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
            d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void flash_attention_nf4(
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
    bool causal_mask = false,
    float* kernel_times = nullptr,
    int* num_kernels = nullptr
) {
    const NF4Block* Wq_blocks = static_cast<const NF4Block*>(d_Wq_nf4);
    const NF4Block* Wk_blocks = static_cast<const NF4Block*>(d_Wk_nf4);
    const NF4Block* Wv_blocks = static_cast<const NF4Block*>(d_Wv_nf4);

    const int NUM_KERNELS = 5;
    if (num_kernels) *num_kernels = NUM_KERNELS;

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
    
    // Allocate temporary buffers for Q, K, V
    float *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));

    // Grid setup
    int max_d = (d_k > d_v) ? d_k : d_v;
    dim3 block_fused(TILE_SIZE, TILE_SIZE);
    dim3 grid_fused((max_d + TILE_SIZE - 1) / TILE_SIZE,
                    (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                    batch);
    dim3 block_fused_qk(256);
    dim3 grid_fused_qk(seq_len, batch);
    const float scale_factor = rsqrtf((float)d_k);
    size_t shared_bytes = seq_len * sizeof(float);

    if (kernel_times) {
        cudaEvent_t starts[NUM_KERNELS], stops[NUM_KERNELS];
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventCreate(&starts[i]));
            CUDA_CHECK(cudaEventCreate(&stops[i]));
        }
        
        float all_times[TIMING_NUM_ITERATIONS][NUM_KERNELS];
        
        for (int iter = 0; iter < TIMING_NUM_ITERATIONS; iter++) {
            CUDA_CHECK(cudaEventRecord(starts[0]));
            dequantize_nf4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq, total_size_q);
            CUDA_CHECK(cudaEventRecord(stops[0]));
            
            CUDA_CHECK(cudaEventRecord(starts[1]));
            dequantize_nf4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk, total_size_q);
            CUDA_CHECK(cudaEventRecord(stops[1]));
            
            CUDA_CHECK(cudaEventRecord(starts[2]));
            dequantize_nf4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv, total_size_v);
            CUDA_CHECK(cudaEventRecord(stops[2]));
            
            CUDA_CHECK(cudaEventRecord(starts[3]));
            fused_qkv_projection_tiled_kernel<<<grid_fused, block_fused>>>(
                d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv,
                d_Q, d_K, d_V,
                batch, seq_len, d_model, d_k, d_v);
            CUDA_CHECK(cudaEventRecord(stops[3]));
            
            CUDA_CHECK(cudaEventRecord(starts[4]));
            flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
                d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
            CUDA_CHECK(cudaEventRecord(stops[4]));
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            for (int k = 0; k < NUM_KERNELS; k++) {
                CUDA_CHECK(cudaEventElapsedTime(&all_times[iter][k], starts[k], stops[k]));
            }
        }
        
        for (int k = 0; k < NUM_KERNELS; k++) {
            float times_for_kernel[TIMING_NUM_ITERATIONS];
            for (int i = 0; i < TIMING_NUM_ITERATIONS; i++) {
                times_for_kernel[i] = all_times[i][k];
            }
            kernel_times[k] = compute_median_local(times_for_kernel, TIMING_NUM_ITERATIONS);
        }
        
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventDestroy(starts[i]));
            CUDA_CHECK(cudaEventDestroy(stops[i]));
        }
    } else {
        dequantize_nf4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq, total_size_q);
        CUDA_CHECK(cudaGetLastError());
        
        dequantize_nf4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk, total_size_q);
        CUDA_CHECK(cudaGetLastError());
        
        dequantize_nf4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv, total_size_v);
        CUDA_CHECK(cudaGetLastError());
        
        fused_qkv_projection_tiled_kernel<<<grid_fused, block_fused>>>(
            d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv,
            d_Q, d_K, d_V,
            batch, seq_len, d_model, d_k, d_v);
        CUDA_CHECK(cudaGetLastError());

        flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
            d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void flash_attention_nvfp4(
    const float* d_X,
    const void* d_Wq_nvfp4,
    const void* d_Wk_nvfp4,
    const void* d_Wv_nvfp4,
    const NVFP4TensorMeta* d_Wq_meta,
    const NVFP4TensorMeta* d_Wk_meta,
    const NVFP4TensorMeta* d_Wv_meta,
    const float* d_bq,
    const float* d_bk,
    const float* d_bv,
    float* d_output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    bool causal_mask = false,
    float* kernel_times = nullptr,
    int* num_kernels = nullptr
) {
    const NVFP4Block* Wq_blocks = static_cast<const NVFP4Block*>(d_Wq_nvfp4);
    const NVFP4Block* Wk_blocks = static_cast<const NVFP4Block*>(d_Wk_nvfp4);
    const NVFP4Block* Wv_blocks = static_cast<const NVFP4Block*>(d_Wv_nvfp4);

    if (num_kernels) *num_kernels = 5;

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
    
    dequantize_nvfp4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq_meta->global_scale_dec, d_Wq, total_size_q);
    CUDA_CHECK(cudaGetLastError());

    dequantize_nvfp4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk_meta->global_scale_dec, d_Wk, total_size_q);
    CUDA_CHECK(cudaGetLastError());

    dequantize_nvfp4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv_meta->global_scale_dec, d_Wv, total_size_v);
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


// ========== New Baselines ==========

void naive_flash_attention_mxfp4(
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
    bool causal_mask = false,
    float* kernel_times = nullptr,
    int* num_kernels = nullptr
) {
    const MXFP4Block* Wq_blocks = static_cast<const MXFP4Block*>(d_Wq_mxfp4);
    const MXFP4Block* Wk_blocks = static_cast<const MXFP4Block*>(d_Wk_mxfp4);
    const MXFP4Block* Wv_blocks = static_cast<const MXFP4Block*>(d_Wv_mxfp4);

    const int NUM_KERNELS = 7;
    if (num_kernels) *num_kernels = NUM_KERNELS;
    
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
    
    // Allocate temporary buffers for Q, K, V, A (no need for QK anymore!)
    float *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));

    dim3 block_proj(16, 16);
    dim3 grid_q((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    dim3 grid_k((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    dim3 grid_v((d_v + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    
    dim3 block_fused_qk(256);
    dim3 grid_fused_qk(seq_len, batch);
    const float scale_factor = rsqrtf((float)d_k);
    size_t shared_bytes = seq_len * sizeof(float);

    // If timing requested, measure kernel times
    if (kernel_times) {
        cudaEvent_t starts[NUM_KERNELS], stops[NUM_KERNELS];
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventCreate(&starts[i]));
            CUDA_CHECK(cudaEventCreate(&stops[i]));
        }
        
        float all_times[TIMING_NUM_ITERATIONS][NUM_KERNELS];
        
        for (int iter = 0; iter < TIMING_NUM_ITERATIONS; iter++) {
            CUDA_CHECK(cudaEventRecord(starts[0]));
            dequantize_mxfp4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq, total_size_q);
            CUDA_CHECK(cudaEventRecord(stops[0]));
            
            CUDA_CHECK(cudaEventRecord(starts[1]));
            dequantize_mxfp4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk, total_size_q);
            CUDA_CHECK(cudaEventRecord(stops[1]));
            
            CUDA_CHECK(cudaEventRecord(starts[2]));
            dequantize_mxfp4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv, total_size_v);
            CUDA_CHECK(cudaEventRecord(stops[2]));
            
            CUDA_CHECK(cudaEventRecord(starts[3]));
            linear_projection_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
            CUDA_CHECK(cudaEventRecord(stops[3]));
            
            CUDA_CHECK(cudaEventRecord(starts[4]));
            linear_projection_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
            CUDA_CHECK(cudaEventRecord(stops[4]));
            
            CUDA_CHECK(cudaEventRecord(starts[5]));
            linear_projection_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
            CUDA_CHECK(cudaEventRecord(stops[5]));
            
            CUDA_CHECK(cudaEventRecord(starts[6]));
            flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
                d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
            CUDA_CHECK(cudaEventRecord(stops[6]));
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            for (int k = 0; k < NUM_KERNELS; k++) {
                CUDA_CHECK(cudaEventElapsedTime(&all_times[iter][k], starts[k], stops[k]));
            }
        }
        
        for (int k = 0; k < NUM_KERNELS; k++) {
            float times_for_kernel[TIMING_NUM_ITERATIONS];
            for (int i = 0; i < TIMING_NUM_ITERATIONS; i++) {
                times_for_kernel[i] = all_times[i][k];
            }
            kernel_times[k] = compute_median_local(times_for_kernel, TIMING_NUM_ITERATIONS);
        }
        
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventDestroy(starts[i]));
            CUDA_CHECK(cudaEventDestroy(stops[i]));
        }
    } else {
        // Normal execution without timing
        dequantize_mxfp4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq, total_size_q);
        CUDA_CHECK(cudaGetLastError());
        
        dequantize_mxfp4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk, total_size_q);
        CUDA_CHECK(cudaGetLastError());
        
        dequantize_mxfp4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv, total_size_v);
        CUDA_CHECK(cudaGetLastError());
        
        linear_projection_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
        CUDA_CHECK(cudaGetLastError());

        linear_projection_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
        CUDA_CHECK(cudaGetLastError());

        linear_projection_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
        CUDA_CHECK(cudaGetLastError());

        flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
            d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void naive_flash_attention_nf4(
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
    bool causal_mask = false,
    float* kernel_times = nullptr,
    int* num_kernels = nullptr
) {
    const NF4Block* Wq_blocks = static_cast<const NF4Block*>(d_Wq_nf4);
    const NF4Block* Wk_blocks = static_cast<const NF4Block*>(d_Wk_nf4);
    const NF4Block* Wv_blocks = static_cast<const NF4Block*>(d_Wv_nf4);

    const int NUM_KERNELS = 7;
    if (num_kernels) *num_kernels = NUM_KERNELS;

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
    
    // Allocate temporary buffers for Q, K, V
    float *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));

    dim3 block_proj(16, 16);
    dim3 grid_q((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    dim3 grid_k((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    dim3 grid_v((d_v + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    
    dim3 block_fused_qk(256);
    dim3 grid_fused_qk(seq_len, batch);
    const float scale_factor = rsqrtf((float)d_k);
    size_t shared_bytes = seq_len * sizeof(float);

    if (kernel_times) {
        cudaEvent_t starts[NUM_KERNELS], stops[NUM_KERNELS];
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventCreate(&starts[i]));
            CUDA_CHECK(cudaEventCreate(&stops[i]));
        }
        
        float all_times[TIMING_NUM_ITERATIONS][NUM_KERNELS];
        
        for (int iter = 0; iter < TIMING_NUM_ITERATIONS; iter++) {
            CUDA_CHECK(cudaEventRecord(starts[0]));
            dequantize_nf4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq, total_size_q);
            CUDA_CHECK(cudaEventRecord(stops[0]));
            
            CUDA_CHECK(cudaEventRecord(starts[1]));
            dequantize_nf4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk, total_size_q);
            CUDA_CHECK(cudaEventRecord(stops[1]));
            
            CUDA_CHECK(cudaEventRecord(starts[2]));
            dequantize_nf4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv, total_size_v);
            CUDA_CHECK(cudaEventRecord(stops[2]));
            
            CUDA_CHECK(cudaEventRecord(starts[3]));
            linear_projection_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
            CUDA_CHECK(cudaEventRecord(stops[3]));
            
            CUDA_CHECK(cudaEventRecord(starts[4]));
            linear_projection_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
            CUDA_CHECK(cudaEventRecord(stops[4]));
            
            CUDA_CHECK(cudaEventRecord(starts[5]));
            linear_projection_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
            CUDA_CHECK(cudaEventRecord(stops[5]));
            
            CUDA_CHECK(cudaEventRecord(starts[6]));
            flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
                d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
            CUDA_CHECK(cudaEventRecord(stops[6]));
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            for (int k = 0; k < NUM_KERNELS; k++) {
                CUDA_CHECK(cudaEventElapsedTime(&all_times[iter][k], starts[k], stops[k]));
            }
        }
        
        for (int k = 0; k < NUM_KERNELS; k++) {
            float times_for_kernel[TIMING_NUM_ITERATIONS];
            for (int i = 0; i < TIMING_NUM_ITERATIONS; i++) {
                times_for_kernel[i] = all_times[i][k];
            }
            kernel_times[k] = compute_median_local(times_for_kernel, TIMING_NUM_ITERATIONS);
        }
        
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventDestroy(starts[i]));
            CUDA_CHECK(cudaEventDestroy(stops[i]));
        }
    } else {
        dequantize_nf4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq, total_size_q);
        CUDA_CHECK(cudaGetLastError());
        
        dequantize_nf4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk, total_size_q);
        CUDA_CHECK(cudaGetLastError());
        
        dequantize_nf4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv, total_size_v);
        CUDA_CHECK(cudaGetLastError());
        
        linear_projection_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
        CUDA_CHECK(cudaGetLastError());

        linear_projection_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
        CUDA_CHECK(cudaGetLastError());

        linear_projection_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
        CUDA_CHECK(cudaGetLastError());

        flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
            d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void naive_flash_attention_nvfp4(
    const float* d_X,
    const void* d_Wq_nvfp4,
    const void* d_Wk_nvfp4,
    const void* d_Wv_nvfp4,
    const NVFP4TensorMeta* d_Wq_meta,
    const NVFP4TensorMeta* d_Wk_meta,
    const NVFP4TensorMeta* d_Wv_meta,
    const float* d_bq,
    const float* d_bk,
    const float* d_bv,
    float* d_output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    bool causal_mask = false,
    float* kernel_times = nullptr,
    int* num_kernels = nullptr
) {
    const NVFP4Block* Wq_blocks = static_cast<const NVFP4Block*>(d_Wq_nvfp4);
    const NVFP4Block* Wk_blocks = static_cast<const NVFP4Block*>(d_Wk_nvfp4);
    const NVFP4Block* Wv_blocks = static_cast<const NVFP4Block*>(d_Wv_nvfp4);

    const int NUM_KERNELS = 7;
    if (num_kernels) *num_kernels = NUM_KERNELS;

    // Allocate temporary buffers for dequantized weights
    float *d_Wq, *d_Wk, *d_Wv;
    CUDA_CHECK(cudaMalloc(&d_Wq, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv, d_model * d_v * sizeof(float)));

    // Dequantize NVFP4 weights
    int total_size_q = d_model * d_k;
    int total_size_v = d_model * d_v;
    
    dim3 block_dequant(256);
    dim3 grid_wq((total_size_q + block_dequant.x - 1) / block_dequant.x);
    dim3 grid_wk((total_size_q + block_dequant.x - 1) / block_dequant.x);
    dim3 grid_wv((total_size_v + block_dequant.x - 1) / block_dequant.x);
    
    // Allocate temporary buffers for Q, K, V
    float *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));

    dim3 block_proj(16, 16);
    dim3 grid_q((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    dim3 grid_k((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    dim3 grid_v((d_v + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    
    dim3 block_fused_qk(256);
    dim3 grid_fused_qk(seq_len, batch);
    const float scale_factor = rsqrtf((float)d_k);
    size_t shared_bytes = seq_len * sizeof(float);

    if (kernel_times) {
        cudaEvent_t starts[NUM_KERNELS], stops[NUM_KERNELS];
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventCreate(&starts[i]));
            CUDA_CHECK(cudaEventCreate(&stops[i]));
        }
        
        float all_times[TIMING_NUM_ITERATIONS][NUM_KERNELS];
        
        for (int iter = 0; iter < TIMING_NUM_ITERATIONS; iter++) {
            CUDA_CHECK(cudaEventRecord(starts[0]));
            dequantize_nvfp4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq_meta->global_scale_dec, d_Wq, total_size_q);
            CUDA_CHECK(cudaEventRecord(stops[0]));
            
            CUDA_CHECK(cudaEventRecord(starts[1]));
            dequantize_nvfp4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk_meta->global_scale_dec, d_Wk, total_size_q);
            CUDA_CHECK(cudaEventRecord(stops[1]));
            
            CUDA_CHECK(cudaEventRecord(starts[2]));
            dequantize_nvfp4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv_meta->global_scale_dec, d_Wv, total_size_v);
            CUDA_CHECK(cudaEventRecord(stops[2]));
            
            CUDA_CHECK(cudaEventRecord(starts[3]));
            linear_projection_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
            CUDA_CHECK(cudaEventRecord(stops[3]));
            
            CUDA_CHECK(cudaEventRecord(starts[4]));
            linear_projection_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
            CUDA_CHECK(cudaEventRecord(stops[4]));
            
            CUDA_CHECK(cudaEventRecord(starts[5]));
            linear_projection_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
            CUDA_CHECK(cudaEventRecord(stops[5]));
            
            CUDA_CHECK(cudaEventRecord(starts[6]));
            flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
                d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
            CUDA_CHECK(cudaEventRecord(stops[6]));
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            for (int k = 0; k < NUM_KERNELS; k++) {
                CUDA_CHECK(cudaEventElapsedTime(&all_times[iter][k], starts[k], stops[k]));
            }
        }
        
        for (int k = 0; k < NUM_KERNELS; k++) {
            float times_for_kernel[TIMING_NUM_ITERATIONS];
            for (int i = 0; i < TIMING_NUM_ITERATIONS; i++) {
                times_for_kernel[i] = all_times[i][k];
            }
            kernel_times[k] = compute_median_local(times_for_kernel, TIMING_NUM_ITERATIONS);
        }
        
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventDestroy(starts[i]));
            CUDA_CHECK(cudaEventDestroy(stops[i]));
        }
    } else {
        dequantize_nvfp4_kernel<<<grid_wq, block_dequant>>>(Wq_blocks, d_Wq_meta->global_scale_dec, d_Wq, total_size_q);
        CUDA_CHECK(cudaGetLastError());

        dequantize_nvfp4_kernel<<<grid_wk, block_dequant>>>(Wk_blocks, d_Wk_meta->global_scale_dec, d_Wk, total_size_q);
        CUDA_CHECK(cudaGetLastError());

        dequantize_nvfp4_kernel<<<grid_wv, block_dequant>>>(Wv_blocks, d_Wv_meta->global_scale_dec, d_Wv, total_size_v);
        CUDA_CHECK(cudaGetLastError());
        
        linear_projection_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
        CUDA_CHECK(cudaGetLastError());

        linear_projection_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
        CUDA_CHECK(cudaGetLastError());

        linear_projection_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
        CUDA_CHECK(cudaGetLastError());

        flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
            d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void fused_flash_attention_mxfp4(
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
    bool causal_mask = false,
    float* kernel_times = nullptr,
    int* num_kernels = nullptr
) {
    const MXFP4Block* Wq_blocks = static_cast<const MXFP4Block*>(d_Wq_mxfp4);
    const MXFP4Block* Wk_blocks = static_cast<const MXFP4Block*>(d_Wk_mxfp4);
    const MXFP4Block* Wv_blocks = static_cast<const MXFP4Block*>(d_Wv_mxfp4);

    const int NUM_KERNELS = 3;
    if (num_kernels) *num_kernels = NUM_KERNELS;

    // Allocate temporary buffers for dequantized weights
    float *d_Wq, *d_Wk, *d_Wv;
    CUDA_CHECK(cudaMalloc(&d_Wq, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv, d_model * d_v * sizeof(float)));

    // Allocate temporary buffers for Q, K, V, A (no need for QK anymore!)
    float *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));

    // Setup grids
    dim3 block_dequant(16, 16);
    dim3 grid_wq((d_k + block_dequant.x - 1) / block_dequant.x,
                 (d_model + block_dequant.y - 1) / block_dequant.y);
    
    int max_d = (d_k > d_v) ? d_k : d_v;
    dim3 block_proj(TILE_SIZE, TILE_SIZE);
    dim3 grid_proj((max_d + TILE_SIZE - 1) / TILE_SIZE,
                   (seq_len + TILE_SIZE - 1) / TILE_SIZE,
                   batch);
    
    dim3 block_fused_qk(256);
    dim3 grid_fused_qk(seq_len, batch);
    const float scale_factor = rsqrtf((float)d_k);
    size_t shared_bytes = seq_len * sizeof(float);

    if (kernel_times) {
        cudaEvent_t starts[NUM_KERNELS], stops[NUM_KERNELS];
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventCreate(&starts[i]));
            CUDA_CHECK(cudaEventCreate(&stops[i]));
        }
        
        float all_times[TIMING_NUM_ITERATIONS][NUM_KERNELS];
        
        for (int iter = 0; iter < TIMING_NUM_ITERATIONS; iter++) {
            CUDA_CHECK(cudaEventRecord(starts[0]));
            fused_mxfp4_qkv_dequant_kernel<<<grid_wq, block_dequant>>>(
                Wq_blocks, Wk_blocks, Wv_blocks,
                d_Wq, d_Wk, d_Wv,
                d_model, d_k, d_v);
            CUDA_CHECK(cudaEventRecord(stops[0]));
            
            CUDA_CHECK(cudaEventRecord(starts[1]));
            fused_qkv_projection_tiled_kernel<<<grid_proj, block_proj>>>(
                d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv,
                d_Q, d_K, d_V,
                batch, seq_len, d_model, d_k, d_v);
            CUDA_CHECK(cudaEventRecord(stops[1]));
            
            CUDA_CHECK(cudaEventRecord(starts[2]));
            flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
                d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
            CUDA_CHECK(cudaEventRecord(stops[2]));
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            for (int k = 0; k < NUM_KERNELS; k++) {
                CUDA_CHECK(cudaEventElapsedTime(&all_times[iter][k], starts[k], stops[k]));
            }
        }
        
        for (int k = 0; k < NUM_KERNELS; k++) {
            float times_for_kernel[TIMING_NUM_ITERATIONS];
            for (int i = 0; i < TIMING_NUM_ITERATIONS; i++) {
                times_for_kernel[i] = all_times[i][k];
            }
            kernel_times[k] = compute_median_local(times_for_kernel, TIMING_NUM_ITERATIONS);
        }
        
        for (int i = 0; i < NUM_KERNELS; i++) {
            CUDA_CHECK(cudaEventDestroy(starts[i]));
            CUDA_CHECK(cudaEventDestroy(stops[i]));
        }
    } else {
        fused_mxfp4_qkv_dequant_kernel<<<grid_wq, block_dequant>>>(
            Wq_blocks, Wk_blocks, Wv_blocks,
            d_Wq, d_Wk, d_Wv,
            d_model, d_k, d_v);
        CUDA_CHECK(cudaGetLastError());

        fused_qkv_projection_tiled_kernel<<<grid_proj, block_proj>>>(
            d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv,
            d_Q, d_K, d_V,
            batch, seq_len, d_model, d_k, d_v);
        CUDA_CHECK(cudaGetLastError());

        flash_attention_kernel<<<grid_fused_qk, block_fused_qk, shared_bytes>>>(
            d_Q, d_K, d_V, d_output, batch, seq_len, d_k, d_v, scale_factor, causal_mask);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void fused_flash_attention_nf4(
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
    bool causal_mask = false,
    float* kernel_times = nullptr,
    int* num_kernels = nullptr
) {
    const NF4Block* Wq_blocks = static_cast<const NF4Block*>(d_Wq_nf4);
    const NF4Block* Wk_blocks = static_cast<const NF4Block*>(d_Wk_nf4);
    const NF4Block* Wv_blocks = static_cast<const NF4Block*>(d_Wv_nf4);

    if (num_kernels) *num_kernels = 3;

    // Allocate temporary buffers for dequantized weights
    float *d_Wq, *d_Wk, *d_Wv;
    CUDA_CHECK(cudaMalloc(&d_Wq, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv, d_model * d_v * sizeof(float)));

    // Step 0: Dequantize MXFP4 weights using fused kernel
    dim3 block_dequant(16, 16);
    dim3 grid_wq((d_k + block_dequant.x - 1) / block_dequant.x,
                 (d_model + block_dequant.y - 1) / block_dequant.y);
    dim3 grid_wv((d_v + block_dequant.x - 1) / block_dequant.x,
                 (d_model + block_dequant.y - 1) / block_dequant.y);
    
    fused_nf4_qkv_dequant_kernel<<<grid_wq, block_dequant>>>(
        Wq_blocks, Wk_blocks, Wv_blocks,
        d_Wq, d_Wk, d_Wv,
        d_model, d_k, d_v);
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

void fused_flash_attention_nvfp4(
    const float* d_X,
    const void* d_Wq_nvfp4,
    const void* d_Wk_nvfp4,
    const void* d_Wv_nvfp4,
    const NVFP4TensorMeta* d_Wq_meta,
    const NVFP4TensorMeta* d_Wk_meta,
    const NVFP4TensorMeta* d_Wv_meta,
    const float* d_bq,
    const float* d_bk,
    const float* d_bv,
    float* d_output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    bool causal_mask = false,
    float* kernel_times = nullptr,
    int* num_kernels = nullptr
) {
    const NVFP4Block* Wq_blocks = static_cast<const NVFP4Block*>(d_Wq_nvfp4);
    const NVFP4Block* Wk_blocks = static_cast<const NVFP4Block*>(d_Wk_nvfp4);
    const NVFP4Block* Wv_blocks = static_cast<const NVFP4Block*>(d_Wv_nvfp4);

    if (num_kernels) *num_kernels = 3;

    // Allocate temporary buffers for dequantized weights
    float *d_Wq, *d_Wk, *d_Wv;
    CUDA_CHECK(cudaMalloc(&d_Wq, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk, d_model * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv, d_model * d_v * sizeof(float)));

    // Step 0: Dequantize MXFP4 weights using fused kernel
    dim3 block_dequant(16, 16);
    dim3 grid_wq((d_k + block_dequant.x - 1) / block_dequant.x,
                 (d_model + block_dequant.y - 1) / block_dequant.y);
    dim3 grid_wv((d_v + block_dequant.x - 1) / block_dequant.x,
                 (d_model + block_dequant.y - 1) / block_dequant.y);
    
    fused_nvfp4_qkv_dequant_kernel<<<grid_wq, block_dequant>>>(
        Wq_blocks, Wk_blocks, Wv_blocks,
        d_Wq_meta->global_scale_dec, d_Wk_meta->global_scale_dec, d_Wv_meta->global_scale_dec,
        d_Wq, d_Wk, d_Wv,
        d_model, d_k, d_v);
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

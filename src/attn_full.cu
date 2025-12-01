#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "quantization_utils.cuh"
#include "attn_full.cuh"

#define WARP_SIZE 32
#define TILE_D 32  // Tile size for d_k/d_v dimension
#define CHUNK_SIZE 32  // Chunk size for K/V tiling to reduce redundant computation

// NF4 lookup table in constant memory (for GPU)
__constant__ float NF4_TABLE_CONST[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

// NVFP4 E2M1 lookup table in constant memory
__constant__ float NVFP4_TABLE_CONST[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Helper device functions for dequantization

// MXFP4 dequantization helper
__device__ inline float fp4_to_float_dev(uint8_t fp4_val, int shared_exp) {
    uint8_t sign = (fp4_val >> 3) & 0x1;
    uint8_t exp_bits = (fp4_val >> 1) & 0x3;
    uint8_t mantissa_bit = fp4_val & 0x1;
    
    if (exp_bits == 0 && mantissa_bit == 0) {
        return 0.0f;
    }
    
    int relative_exp;
    uint32_t mantissa;
    
    if (exp_bits == 0) {
        relative_exp = -1;
        mantissa = (1 << 23);
    } else {
        relative_exp = (int)exp_bits - 1;
        mantissa = (1 << 23) | (mantissa_bit << 22);
    }
    
    int actual_exp = shared_exp + relative_exp;
    uint32_t result_bits = (sign << 31) | ((actual_exp + 127) << 23) | (mantissa & 0x7FFFFF);
    return *reinterpret_cast<float*>(&result_bits);
}

// FP8 E4M3 to float (for NVFP4 scales)
__device__ inline float fp8_e4m3_to_float_dev(uint8_t fp8_val) {
    uint8_t sign = (fp8_val >> 7) & 0x1;
    uint8_t exp = (fp8_val >> 3) & 0xF;
    uint8_t mantissa = fp8_val & 0x7;
    
    if (exp == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        float val = mantissa / 8.0f;
        val *= powf(2.0f, -6.0f);
        return sign ? -val : val;
    }
    
    float val = 1.0f + mantissa / 8.0f;
    val *= powf(2.0f, (int)exp - 7);
    return sign ? -val : val;
}

// ============================================================================
// Blockwise INT8 Quantization - Full Fusion Kernel
// ============================================================================

__global__ void full_attention_blockwise_kernel(
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
    float* __restrict__ output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    int block_size,
    float scale_factor,
    bool causal_mask
) {
    int b = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (b >= batch || row >= seq_len) return;
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Shared memory layout (optimized for chunk-based tiling)
    extern __shared__ float shared_mem[];
    float* q_row = shared_mem;  // Size: d_k
    int max_kv_size = (d_k > d_v ? d_k : d_v) * CHUNK_SIZE;
    float* kv_tile = shared_mem + d_k;  // Size: max(d_k, d_v) * CHUNK_SIZE (reused for K and V)
    float* logits = kv_tile + max_kv_size;  // Size: seq_len
    float* warp_maxes = logits + seq_len;  // Size: 32
    float* warp_sums = warp_maxes + 32;  // Size: 32
    
    // Base indices
    int x_base = b * seq_len * d_model + row * d_model;
    int out_base = b * seq_len * d_v + row * d_v;
    
    // Step 1: Compute Q[row, :] = X[row, :] × Wq + bq
    for (int k = tid; k < d_k; k += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < d_model; ++d) {
            int w_idx = d * d_k + k;
            int block_idx = w_idx / block_size;
            
            float scale = Wq_scales[block_idx];
            int8_t zero = Wq_zeros[block_idx];
            int8_t qval = Wq_quantized[w_idx];
            float w_dequant = scale * (float)(qval - zero);
            
            sum += X[x_base + d] * w_dequant;
        }
        q_row[k] = sum + bq[k];
    }
    __syncthreads();
    
    // Step 2: Initialize logits
    for (int col = tid; col < seq_len; col += blockDim.x) {
        logits[col] = 0.0f;
    }
    __syncthreads();
    
    // Step 3: Compute QK^T using chunk-based tiling
    // Process seq_len in chunks to reuse K computation
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = (chunk_start + CHUNK_SIZE < seq_len) ? (chunk_start + CHUNK_SIZE) : seq_len;
        int chunk_size = chunk_end - chunk_start;
        
        // Phase 3.1: Cooperatively compute K tile for this chunk
        // K_tile[chunk_idx, k] = X[chunk_start + chunk_idx, :] × Wk[:, k] + bk[k]
        for (int idx = tid; idx < chunk_size * d_k; idx += blockDim.x) {
            int chunk_idx = idx / d_k;
            int k = idx % d_k;
            int col = chunk_start + chunk_idx;
            int x_col_base = b * seq_len * d_model + col * d_model;
            
            float k_val = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                int w_idx = d * d_k + k;
                int block_idx = w_idx / block_size;
                
                float scale = Wk_scales[block_idx];
                int8_t zero = Wk_zeros[block_idx];
                int8_t qval = Wk_quantized[w_idx];
                float w_dequant = scale * (float)(qval - zero);
                
                k_val += X[x_col_base + d] * w_dequant;
            }
            kv_tile[chunk_idx * d_k + k] = k_val + bk[k];
        }
        __syncthreads();
        
        // Phase 3.2: Compute Q · K_tile^T for this chunk
        for (int chunk_idx = tid; chunk_idx < chunk_size; chunk_idx += blockDim.x) {
            float dot_product = 0.0f;
            for (int k = 0; k < d_k; ++k) {
                dot_product += q_row[k] * kv_tile[chunk_idx * d_k + k];
            }
            logits[chunk_start + chunk_idx] = dot_product * scale_factor;
        }
        __syncthreads();
    }
    
    // Step 4: Apply causal mask and compute max for softmax
    float thread_max = -INFINITY;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = logits[col];
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
    if (lane_id == 0) {
        warp_maxes[warp_id] = thread_max;
    }
    __syncthreads();
    
    // Final max reduction
    float global_max = -INFINITY;
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        global_max = warp_maxes[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        global_max = fmaxf(global_max, __shfl_down_sync(0xffffffff, global_max, offset));
    }
    if (warp_id == 0 && lane_id == 0) {
        warp_maxes[0] = global_max;
    }
    __syncthreads();
    global_max = warp_maxes[0];
    
    // Step 5: Compute exp and sum
    float thread_sum = 0.0f;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = logits[col];
        if (causal_mask && col > row) {
            logits[col] = 0.0f;
        } else {
            float exp_val = expf(val - global_max);
            logits[col] = exp_val;
            thread_sum += exp_val;
        }
    }
    
    // Warp-level sum reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final sum reduction
    float global_sum = 0.0f;
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        global_sum = warp_sums[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        global_sum += __shfl_down_sync(0xffffffff, global_sum, offset);
    }
    if (warp_id == 0 && lane_id == 0) {
        warp_sums[0] = global_sum;
    }
    __syncthreads();
    global_sum = warp_sums[0];
    
    // Step 6: Normalize softmax
    if (global_sum > 0.0f) {
        for (int col = tid; col < seq_len; col += blockDim.x) {
            logits[col] /= global_sum;
        }
    }
    __syncthreads();
    
    // Step 7: Compute output = A × V using chunk-based tiling
    // Initialize output to zero
    for (int v_dim = tid; v_dim < d_v; v_dim += blockDim.x) {
        output[out_base + v_dim] = 0.0f;
    }
    __syncthreads();
    
    // Process seq_len in chunks to reuse V computation
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = (chunk_start + CHUNK_SIZE < seq_len) ? (chunk_start + CHUNK_SIZE) : seq_len;
        int chunk_size = chunk_end - chunk_start;
        
        // Phase 7.1: Cooperatively compute V tile for this chunk
        // V_tile[chunk_idx, v] = X[chunk_start + chunk_idx, :] × Wv[:, v] + bv[v]
        for (int idx = tid; idx < chunk_size * d_v; idx += blockDim.x) {
            int chunk_idx = idx / d_v;
            int v = idx % d_v;
            int col = chunk_start + chunk_idx;
            int x_col_base = b * seq_len * d_model + col * d_model;
            
            float v_val = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                int w_idx = d * d_v + v;
                int block_idx = w_idx / block_size;
                
                float scale = Wv_scales[block_idx];
                int8_t zero = Wv_zeros[block_idx];
                int8_t qval = Wv_quantized[w_idx];
                float w_dequant = scale * (float)(qval - zero);
                
                v_val += X[x_col_base + d] * w_dequant;
            }
            kv_tile[chunk_idx * d_v + v] = v_val + bv[v];
        }
        __syncthreads();
        
        // Phase 7.2: Compute A[chunk] · V_tile and accumulate to output
        for (int v_dim = tid; v_dim < d_v; v_dim += blockDim.x) {
            float acc = 0.0f;
            for (int chunk_idx = 0; chunk_idx < chunk_size; ++chunk_idx) {
                float attn_weight = logits[chunk_start + chunk_idx];
                acc += attn_weight * kv_tile[chunk_idx * d_v + v_dim];
            }
            output[out_base + v_dim] += acc;
        }
        __syncthreads();
    }
}

void full_attention_quantized_blockwise(
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
    bool causal_mask
) {
    dim3 grid(seq_len, batch);
    dim3 block(256);
    
    // Shared memory: q_row + kv_tile + logits + warp_reduction
    int max_kv_size = (d_k > d_v ? d_k : d_v) * CHUNK_SIZE;
    size_t shared_bytes = (d_k + max_kv_size + seq_len + 64) * sizeof(float);
    const float scale_factor = rsqrtf((float)d_k);
    
    full_attention_blockwise_kernel<<<grid, block, shared_bytes>>>(
        d_X, d_Wq_quantized, d_Wk_quantized, d_Wv_quantized,
        d_Wq_scales, d_Wk_scales, d_Wv_scales,
        d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
        d_bq, d_bk, d_bv, d_output,
        batch, seq_len, d_model, d_k, d_v, block_size,
        scale_factor, causal_mask
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// MXFP4 - Full Fusion Kernel
// ============================================================================

__global__ void full_attention_mxfp4_kernel(
    const float* __restrict__ X,
    const MXFP4Block* __restrict__ Wq_blocks,
    const MXFP4Block* __restrict__ Wk_blocks,
    const MXFP4Block* __restrict__ Wv_blocks,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv,
    float* __restrict__ output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    float scale_factor,
    bool causal_mask
) {
    int b = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (b >= batch || row >= seq_len) return;
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    extern __shared__ float shared_mem[];
    float* q_row = shared_mem;
    int max_kv_size = (d_k > d_v ? d_k : d_v) * CHUNK_SIZE;
    float* kv_tile = shared_mem + d_k;
    float* logits = kv_tile + max_kv_size;
    float* warp_maxes = logits + seq_len;
    float* warp_sums = warp_maxes + 32;
    
    int x_base = b * seq_len * d_model + row * d_model;
    int out_base = b * seq_len * d_v + row * d_v;
    
    // Step 1: Compute Q[row, :]
    for (int k = tid; k < d_k; k += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < d_model; ++d) {
            int w_idx = d * d_k + k;
            int block_idx = w_idx / MXFP4_BLOCK_SIZE;
            int local_idx = w_idx % MXFP4_BLOCK_SIZE;
            
            int shared_exp = (int)Wq_blocks[block_idx].shared_exp - 127;
            uint8_t fp4_val;
            if (local_idx % 2 == 0) {
                fp4_val = (Wq_blocks[block_idx].data[local_idx / 2] >> 4) & 0xF;
            } else {
                fp4_val = Wq_blocks[block_idx].data[local_idx / 2] & 0xF;
            }
            float w_dequant = fp4_to_float_dev(fp4_val, shared_exp);
            
            sum += X[x_base + d] * w_dequant;
        }
        q_row[k] = sum + bq[k];
    }
    __syncthreads();
    
    // Step 2: Initialize logits
    for (int col = tid; col < seq_len; col += blockDim.x) {
        logits[col] = 0.0f;
    }
    __syncthreads();
    
    // Step 3: Compute QK^T using chunk-based tiling
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = (chunk_start + CHUNK_SIZE < seq_len) ? (chunk_start + CHUNK_SIZE) : seq_len;
        int chunk_size = chunk_end - chunk_start;
        
        // Phase 3.1: Compute K tile
        for (int idx = tid; idx < chunk_size * d_k; idx += blockDim.x) {
            int chunk_idx = idx / d_k;
            int k = idx % d_k;
            int col = chunk_start + chunk_idx;
            int x_col_base = b * seq_len * d_model + col * d_model;
            
            float k_val = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                int w_idx = d * d_k + k;
                int block_idx = w_idx / MXFP4_BLOCK_SIZE;
                int local_idx = w_idx % MXFP4_BLOCK_SIZE;
                
                int shared_exp = (int)Wk_blocks[block_idx].shared_exp - 127;
                uint8_t fp4_val;
                if (local_idx % 2 == 0) {
                    fp4_val = (Wk_blocks[block_idx].data[local_idx / 2] >> 4) & 0xF;
                } else {
                    fp4_val = Wk_blocks[block_idx].data[local_idx / 2] & 0xF;
                }
                float w_dequant = fp4_to_float_dev(fp4_val, shared_exp);
                
                k_val += X[x_col_base + d] * w_dequant;
            }
            kv_tile[chunk_idx * d_k + k] = k_val + bk[k];
        }
        __syncthreads();
        
        // Phase 3.2: Compute Q · K_tile^T
        for (int chunk_idx = tid; chunk_idx < chunk_size; chunk_idx += blockDim.x) {
            float dot_product = 0.0f;
            for (int k = 0; k < d_k; ++k) {
                dot_product += q_row[k] * kv_tile[chunk_idx * d_k + k];
            }
            logits[chunk_start + chunk_idx] = dot_product * scale_factor;
        }
        __syncthreads();
    }
    
    // Step 4: Softmax - max
    float thread_max = -INFINITY;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = logits[col];
        if (causal_mask && col > row) {
            val = -INFINITY;
        }
        logits[col] = val;
        thread_max = fmaxf(thread_max, val);
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    if (lane_id == 0) {
        warp_maxes[warp_id] = thread_max;
    }
    __syncthreads();
    
    float global_max = -INFINITY;
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        global_max = warp_maxes[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        global_max = fmaxf(global_max, __shfl_down_sync(0xffffffff, global_max, offset));
    }
    if (warp_id == 0 && lane_id == 0) {
        warp_maxes[0] = global_max;
    }
    __syncthreads();
    global_max = warp_maxes[0];
    
    // Step 5: Softmax - exp and sum
    float thread_sum = 0.0f;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = logits[col];
        if (causal_mask && col > row) {
            logits[col] = 0.0f;
        } else {
            float exp_val = expf(val - global_max);
            logits[col] = exp_val;
            thread_sum += exp_val;
        }
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    float global_sum = 0.0f;
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        global_sum = warp_sums[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        global_sum += __shfl_down_sync(0xffffffff, global_sum, offset);
    }
    if (warp_id == 0 && lane_id == 0) {
        warp_sums[0] = global_sum;
    }
    __syncthreads();
    global_sum = warp_sums[0];
    
    // Step 6: Normalize
    if (global_sum > 0.0f) {
        for (int col = tid; col < seq_len; col += blockDim.x) {
            logits[col] /= global_sum;
        }
    }
    __syncthreads();
    
    // Step 7: Compute output = A × V using chunk-based tiling
    for (int v_dim = tid; v_dim < d_v; v_dim += blockDim.x) {
        output[out_base + v_dim] = 0.0f;
    }
    __syncthreads();
    
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = (chunk_start + CHUNK_SIZE < seq_len) ? (chunk_start + CHUNK_SIZE) : seq_len;
        int chunk_size = chunk_end - chunk_start;
        
        // Phase 7.1: Compute V tile
        for (int idx = tid; idx < chunk_size * d_v; idx += blockDim.x) {
            int chunk_idx = idx / d_v;
            int v = idx % d_v;
            int col = chunk_start + chunk_idx;
            int x_col_base = b * seq_len * d_model + col * d_model;
            
            float v_val = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                int w_idx = d * d_v + v;
                int block_idx = w_idx / MXFP4_BLOCK_SIZE;
                int local_idx = w_idx % MXFP4_BLOCK_SIZE;
                
                int shared_exp = (int)Wv_blocks[block_idx].shared_exp - 127;
                uint8_t fp4_val;
                if (local_idx % 2 == 0) {
                    fp4_val = (Wv_blocks[block_idx].data[local_idx / 2] >> 4) & 0xF;
                } else {
                    fp4_val = Wv_blocks[block_idx].data[local_idx / 2] & 0xF;
                }
                float w_dequant = fp4_to_float_dev(fp4_val, shared_exp);
                
                v_val += X[x_col_base + d] * w_dequant;
            }
            kv_tile[chunk_idx * d_v + v] = v_val + bv[v];
        }
        __syncthreads();
        
        // Phase 7.2: Compute A[chunk] · V_tile
        for (int v_dim = tid; v_dim < d_v; v_dim += blockDim.x) {
            float acc = 0.0f;
            for (int chunk_idx = 0; chunk_idx < chunk_size; ++chunk_idx) {
                float attn_weight = logits[chunk_start + chunk_idx];
                acc += attn_weight * kv_tile[chunk_idx * d_v + v_dim];
            }
            output[out_base + v_dim] += acc;
        }
        __syncthreads();
    }
}

void full_attention_mxfp4(
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
    bool causal_mask
) {
    const MXFP4Block* Wq_blocks = static_cast<const MXFP4Block*>(d_Wq_mxfp4);
    const MXFP4Block* Wk_blocks = static_cast<const MXFP4Block*>(d_Wk_mxfp4);
    const MXFP4Block* Wv_blocks = static_cast<const MXFP4Block*>(d_Wv_mxfp4);
    
    dim3 grid(seq_len, batch);
    dim3 block(256);
    
    int max_kv_size = (d_k > d_v ? d_k : d_v) * CHUNK_SIZE;
    size_t shared_bytes = (d_k + max_kv_size + seq_len + 64) * sizeof(float);
    const float scale_factor = rsqrtf((float)d_k);
    
    full_attention_mxfp4_kernel<<<grid, block, shared_bytes>>>(
        d_X, Wq_blocks, Wk_blocks, Wv_blocks,
        d_bq, d_bk, d_bv, d_output,
        batch, seq_len, d_model, d_k, d_v,
        scale_factor, causal_mask
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// NF4 - Full Fusion Kernel
// ============================================================================

__global__ void full_attention_nf4_kernel(
    const float* __restrict__ X,
    const NF4Block* __restrict__ Wq_blocks,
    const NF4Block* __restrict__ Wk_blocks,
    const NF4Block* __restrict__ Wv_blocks,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv,
    float* __restrict__ output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    float scale_factor,
    bool causal_mask
) {
    int b = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (b >= batch || row >= seq_len) return;
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    extern __shared__ float shared_mem[];
    float* q_row = shared_mem;
    int max_kv_size = (d_k > d_v ? d_k : d_v) * CHUNK_SIZE;
    float* kv_tile = shared_mem + d_k;
    float* logits = kv_tile + max_kv_size;
    float* warp_maxes = logits + seq_len;
    float* warp_sums = warp_maxes + 32;
    
    int x_base = b * seq_len * d_model + row * d_model;
    int out_base = b * seq_len * d_v + row * d_v;
    
    // Step 1: Compute Q[row, :]
    for (int k = tid; k < d_k; k += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < d_model; ++d) {
            int w_idx = d * d_k + k;
            int block_idx = w_idx / NF4_BLOCK_SIZE;
            int local_idx = w_idx % NF4_BLOCK_SIZE;
            
            float absmax = Wq_blocks[block_idx].absmax;
            uint8_t nf4_val;
            if (local_idx % 2 == 0) {
                nf4_val = (Wq_blocks[block_idx].data[local_idx / 2] >> 4) & 0xF;
            } else {
                nf4_val = Wq_blocks[block_idx].data[local_idx / 2] & 0xF;
            }
            if (nf4_val >= 16) nf4_val = 15;
            float w_dequant = NF4_TABLE_CONST[nf4_val] * absmax;
            
            sum += X[x_base + d] * w_dequant;
        }
        q_row[k] = sum + bq[k];
    }
    __syncthreads();
    
    // Step 2: Initialize logits
    for (int col = tid; col < seq_len; col += blockDim.x) {
        logits[col] = 0.0f;
    }
    __syncthreads();
    
    // Step 3: Compute QK^T using chunk-based tiling
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = (chunk_start + CHUNK_SIZE < seq_len) ? (chunk_start + CHUNK_SIZE) : seq_len;
        int chunk_size = chunk_end - chunk_start;
        
        // Phase 3.1: Compute K tile
        for (int idx = tid; idx < chunk_size * d_k; idx += blockDim.x) {
            int chunk_idx = idx / d_k;
            int k = idx % d_k;
            int col = chunk_start + chunk_idx;
            int x_col_base = b * seq_len * d_model + col * d_model;
            
            float k_val = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                int w_idx = d * d_k + k;
                int block_idx = w_idx / NF4_BLOCK_SIZE;
                int local_idx = w_idx % NF4_BLOCK_SIZE;
                
                float absmax = Wk_blocks[block_idx].absmax;
                uint8_t nf4_val;
                if (local_idx % 2 == 0) {
                    nf4_val = (Wk_blocks[block_idx].data[local_idx / 2] >> 4) & 0xF;
                } else {
                    nf4_val = Wk_blocks[block_idx].data[local_idx / 2] & 0xF;
                }
                if (nf4_val >= 16) nf4_val = 15;
                float w_dequant = NF4_TABLE_CONST[nf4_val] * absmax;
                
                k_val += X[x_col_base + d] * w_dequant;
            }
            kv_tile[chunk_idx * d_k + k] = k_val + bk[k];
        }
        __syncthreads();
        
        // Phase 3.2: Compute Q · K_tile^T
        for (int chunk_idx = tid; chunk_idx < chunk_size; chunk_idx += blockDim.x) {
            float dot_product = 0.0f;
            for (int k = 0; k < d_k; ++k) {
                dot_product += q_row[k] * kv_tile[chunk_idx * d_k + k];
            }
            logits[chunk_start + chunk_idx] = dot_product * scale_factor;
        }
        __syncthreads();
    }
    
    // Step 4: Softmax - max
    float thread_max = -INFINITY;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = logits[col];
        if (causal_mask && col > row) {
            val = -INFINITY;
        }
        logits[col] = val;
        thread_max = fmaxf(thread_max, val);
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    if (lane_id == 0) {
        warp_maxes[warp_id] = thread_max;
    }
    __syncthreads();
    
    float global_max = -INFINITY;
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        global_max = warp_maxes[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        global_max = fmaxf(global_max, __shfl_down_sync(0xffffffff, global_max, offset));
    }
    if (warp_id == 0 && lane_id == 0) {
        warp_maxes[0] = global_max;
    }
    __syncthreads();
    global_max = warp_maxes[0];
    
    // Step 5: Softmax - exp and sum
    float thread_sum = 0.0f;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = logits[col];
        if (causal_mask && col > row) {
            logits[col] = 0.0f;
        } else {
            float exp_val = expf(val - global_max);
            logits[col] = exp_val;
            thread_sum += exp_val;
        }
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    float global_sum = 0.0f;
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        global_sum = warp_sums[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        global_sum += __shfl_down_sync(0xffffffff, global_sum, offset);
    }
    if (warp_id == 0 && lane_id == 0) {
        warp_sums[0] = global_sum;
    }
    __syncthreads();
    global_sum = warp_sums[0];
    
    // Step 6: Normalize
    if (global_sum > 0.0f) {
        for (int col = tid; col < seq_len; col += blockDim.x) {
            logits[col] /= global_sum;
        }
    }
    __syncthreads();
    
    // Step 7: Compute output = A × V using chunk-based tiling
    for (int v_dim = tid; v_dim < d_v; v_dim += blockDim.x) {
        output[out_base + v_dim] = 0.0f;
    }
    __syncthreads();
    
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = (chunk_start + CHUNK_SIZE < seq_len) ? (chunk_start + CHUNK_SIZE) : seq_len;
        int chunk_size = chunk_end - chunk_start;
        
        // Phase 7.1: Compute V tile
        for (int idx = tid; idx < chunk_size * d_v; idx += blockDim.x) {
            int chunk_idx = idx / d_v;
            int v = idx % d_v;
            int col = chunk_start + chunk_idx;
            int x_col_base = b * seq_len * d_model + col * d_model;
            
            float v_val = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                int w_idx = d * d_v + v;
                int block_idx = w_idx / NF4_BLOCK_SIZE;
                int local_idx = w_idx % NF4_BLOCK_SIZE;
                
                float absmax = Wv_blocks[block_idx].absmax;
                uint8_t nf4_val;
                if (local_idx % 2 == 0) {
                    nf4_val = (Wv_blocks[block_idx].data[local_idx / 2] >> 4) & 0xF;
                } else {
                    nf4_val = Wv_blocks[block_idx].data[local_idx / 2] & 0xF;
                }
                if (nf4_val >= 16) nf4_val = 15;
                float w_dequant = NF4_TABLE_CONST[nf4_val] * absmax;
                
                v_val += X[x_col_base + d] * w_dequant;
            }
            kv_tile[chunk_idx * d_v + v] = v_val + bv[v];
        }
        __syncthreads();
        
        // Phase 7.2: Compute A[chunk] · V_tile
        for (int v_dim = tid; v_dim < d_v; v_dim += blockDim.x) {
            float acc = 0.0f;
            for (int chunk_idx = 0; chunk_idx < chunk_size; ++chunk_idx) {
                float attn_weight = logits[chunk_start + chunk_idx];
                acc += attn_weight * kv_tile[chunk_idx * d_v + v_dim];
            }
            output[out_base + v_dim] += acc;
        }
        __syncthreads();
    }
}

void full_attention_nf4(
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
    bool causal_mask
) {
    const NF4Block* Wq_blocks = static_cast<const NF4Block*>(d_Wq_nf4);
    const NF4Block* Wk_blocks = static_cast<const NF4Block*>(d_Wk_nf4);
    const NF4Block* Wv_blocks = static_cast<const NF4Block*>(d_Wv_nf4);
    
    dim3 grid(seq_len, batch);
    dim3 block(256);
    
    int max_kv_size = (d_k > d_v ? d_k : d_v) * CHUNK_SIZE;
    size_t shared_bytes = (d_k + max_kv_size + seq_len + 64) * sizeof(float);
    const float scale_factor = rsqrtf((float)d_k);
    
    full_attention_nf4_kernel<<<grid, block, shared_bytes>>>(
        d_X, Wq_blocks, Wk_blocks, Wv_blocks,
        d_bq, d_bk, d_bv, d_output,
        batch, seq_len, d_model, d_k, d_v,
        scale_factor, causal_mask
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// NVFP4 - Full Fusion Kernel
// ============================================================================

__global__ void full_attention_nvfp4_kernel(
    const float* __restrict__ X,
    const NVFP4Block* __restrict__ Wq_blocks,
    const NVFP4Block* __restrict__ Wk_blocks,
    const NVFP4Block* __restrict__ Wv_blocks,
    float s_dec_global_q,
    float s_dec_global_k,
    float s_dec_global_v,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv,
    float* __restrict__ output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    float scale_factor,
    bool causal_mask
) {
    int b = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (b >= batch || row >= seq_len) return;
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    extern __shared__ float shared_mem[];
    float* q_row = shared_mem;
    int max_kv_size = (d_k > d_v ? d_k : d_v) * CHUNK_SIZE;
    float* kv_tile = shared_mem + d_k;
    float* logits = kv_tile + max_kv_size;
    float* warp_maxes = logits + seq_len;
    float* warp_sums = warp_maxes + 32;
    
    int x_base = b * seq_len * d_model + row * d_model;
    int out_base = b * seq_len * d_v + row * d_v;
    
    // Step 1: Compute Q[row, :]
    for (int k = tid; k < d_k; k += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < d_model; ++d) {
            int w_idx = d * d_k + k;
            int block_idx = w_idx / NVFP4_BLOCK_SIZE;
            int local_idx = w_idx % NVFP4_BLOCK_SIZE;
            
            float s_dec_local_e4m3 = fp8_e4m3_to_float_dev(Wq_blocks[block_idx].scale_e4m3);
            float combined_scale = s_dec_local_e4m3 * s_dec_global_q;
            
            uint8_t nvfp4_val;
            if (local_idx % 2 == 0) {
                nvfp4_val = (Wq_blocks[block_idx].data[local_idx / 2] >> 4) & 0xF;
            } else {
                nvfp4_val = Wq_blocks[block_idx].data[local_idx / 2] & 0xF;
            }
            if (nvfp4_val >= 16) nvfp4_val = 15;
            float w_dequant = NVFP4_TABLE_CONST[nvfp4_val] * combined_scale;
            
            sum += X[x_base + d] * w_dequant;
        }
        q_row[k] = sum + bq[k];
    }
    __syncthreads();
    
    // Step 2: Initialize logits
    for (int col = tid; col < seq_len; col += blockDim.x) {
        logits[col] = 0.0f;
    }
    __syncthreads();
    
    // Step 3: Compute QK^T using chunk-based tiling
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = (chunk_start + CHUNK_SIZE < seq_len) ? (chunk_start + CHUNK_SIZE) : seq_len;
        int chunk_size = chunk_end - chunk_start;
        
        // Phase 3.1: Compute K tile
        for (int idx = tid; idx < chunk_size * d_k; idx += blockDim.x) {
            int chunk_idx = idx / d_k;
            int k = idx % d_k;
            int col = chunk_start + chunk_idx;
            int x_col_base = b * seq_len * d_model + col * d_model;
            
            float k_val = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                int w_idx = d * d_k + k;
                int block_idx = w_idx / NVFP4_BLOCK_SIZE;
                int local_idx = w_idx % NVFP4_BLOCK_SIZE;
                
                float s_dec_local_e4m3 = fp8_e4m3_to_float_dev(Wk_blocks[block_idx].scale_e4m3);
                float combined_scale = s_dec_local_e4m3 * s_dec_global_k;
                
                uint8_t nvfp4_val;
                if (local_idx % 2 == 0) {
                    nvfp4_val = (Wk_blocks[block_idx].data[local_idx / 2] >> 4) & 0xF;
                } else {
                    nvfp4_val = Wk_blocks[block_idx].data[local_idx / 2] & 0xF;
                }
                if (nvfp4_val >= 16) nvfp4_val = 15;
                float w_dequant = NVFP4_TABLE_CONST[nvfp4_val] * combined_scale;
                
                k_val += X[x_col_base + d] * w_dequant;
            }
            kv_tile[chunk_idx * d_k + k] = k_val + bk[k];
        }
        __syncthreads();
        
        // Phase 3.2: Compute Q · K_tile^T
        for (int chunk_idx = tid; chunk_idx < chunk_size; chunk_idx += blockDim.x) {
            float dot_product = 0.0f;
            for (int k = 0; k < d_k; ++k) {
                dot_product += q_row[k] * kv_tile[chunk_idx * d_k + k];
            }
            logits[chunk_start + chunk_idx] = dot_product * scale_factor;
        }
        __syncthreads();
    }
    
    // Step 4: Softmax - max
    float thread_max = -INFINITY;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = logits[col];
        if (causal_mask && col > row) {
            val = -INFINITY;
        }
        logits[col] = val;
        thread_max = fmaxf(thread_max, val);
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    if (lane_id == 0) {
        warp_maxes[warp_id] = thread_max;
    }
    __syncthreads();
    
    float global_max = -INFINITY;
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        global_max = warp_maxes[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        global_max = fmaxf(global_max, __shfl_down_sync(0xffffffff, global_max, offset));
    }
    if (warp_id == 0 && lane_id == 0) {
        warp_maxes[0] = global_max;
    }
    __syncthreads();
    global_max = warp_maxes[0];
    
    // Step 5: Softmax - exp and sum
    float thread_sum = 0.0f;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = logits[col];
        if (causal_mask && col > row) {
            logits[col] = 0.0f;
        } else {
            float exp_val = expf(val - global_max);
            logits[col] = exp_val;
            thread_sum += exp_val;
        }
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    float global_sum = 0.0f;
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        global_sum = warp_sums[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        global_sum += __shfl_down_sync(0xffffffff, global_sum, offset);
    }
    if (warp_id == 0 && lane_id == 0) {
        warp_sums[0] = global_sum;
    }
    __syncthreads();
    global_sum = warp_sums[0];
    
    // Step 6: Normalize
    if (global_sum > 0.0f) {
        for (int col = tid; col < seq_len; col += blockDim.x) {
            logits[col] /= global_sum;
        }
    }
    __syncthreads();
    
    // Step 7: Compute output = A × V using chunk-based tiling
    for (int v_dim = tid; v_dim < d_v; v_dim += blockDim.x) {
        output[out_base + v_dim] = 0.0f;
    }
    __syncthreads();
    
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = (chunk_start + CHUNK_SIZE < seq_len) ? (chunk_start + CHUNK_SIZE) : seq_len;
        int chunk_size = chunk_end - chunk_start;
        
        // Phase 7.1: Compute V tile
        for (int idx = tid; idx < chunk_size * d_v; idx += blockDim.x) {
            int chunk_idx = idx / d_v;
            int v = idx % d_v;
            int col = chunk_start + chunk_idx;
            int x_col_base = b * seq_len * d_model + col * d_model;
            
            float v_val = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                int w_idx = d * d_v + v;
                int block_idx = w_idx / NVFP4_BLOCK_SIZE;
                int local_idx = w_idx % NVFP4_BLOCK_SIZE;
                
                float s_dec_local_e4m3 = fp8_e4m3_to_float_dev(Wv_blocks[block_idx].scale_e4m3);
                float combined_scale = s_dec_local_e4m3 * s_dec_global_v;
                
                uint8_t nvfp4_val;
                if (local_idx % 2 == 0) {
                    nvfp4_val = (Wv_blocks[block_idx].data[local_idx / 2] >> 4) & 0xF;
                } else {
                    nvfp4_val = Wv_blocks[block_idx].data[local_idx / 2] & 0xF;
                }
                if (nvfp4_val >= 16) nvfp4_val = 15;
                float w_dequant = NVFP4_TABLE_CONST[nvfp4_val] * combined_scale;
                
                v_val += X[x_col_base + d] * w_dequant;
            }
            kv_tile[chunk_idx * d_v + v] = v_val + bv[v];
        }
        __syncthreads();
        
        // Phase 7.2: Compute A[chunk] · V_tile
        for (int v_dim = tid; v_dim < d_v; v_dim += blockDim.x) {
            float acc = 0.0f;
            for (int chunk_idx = 0; chunk_idx < chunk_size; ++chunk_idx) {
                float attn_weight = logits[chunk_start + chunk_idx];
                acc += attn_weight * kv_tile[chunk_idx * d_v + v_dim];
            }
            output[out_base + v_dim] += acc;
        }
        __syncthreads();
    }
}

void full_attention_nvfp4(
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
    bool causal_mask
) {
    const NVFP4Block* Wq_blocks = static_cast<const NVFP4Block*>(d_Wq_nvfp4);
    const NVFP4Block* Wk_blocks = static_cast<const NVFP4Block*>(d_Wk_nvfp4);
    const NVFP4Block* Wv_blocks = static_cast<const NVFP4Block*>(d_Wv_nvfp4);
    
    dim3 grid(seq_len, batch);
    dim3 block(256);
    
    int max_kv_size = (d_k > d_v ? d_k : d_v) * CHUNK_SIZE;
    size_t shared_bytes = (d_k + max_kv_size + seq_len + 64) * sizeof(float);
    const float scale_factor = rsqrtf((float)d_k);
    
    full_attention_nvfp4_kernel<<<grid, block, shared_bytes>>>(
        d_X, Wq_blocks, Wk_blocks, Wv_blocks,
        d_Wq_meta->global_scale_dec, d_Wk_meta->global_scale_dec, d_Wv_meta->global_scale_dec,
        d_bq, d_bk, d_bv, d_output,
        batch, seq_len, d_model, d_k, d_v,
        scale_factor, causal_mask
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


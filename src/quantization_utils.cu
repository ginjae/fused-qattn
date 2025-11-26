#include "quantization_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cfloat>

// Helper function to simulate block-wise quantization
void quantize_blockwise(
    const float* weights,
    int8_t* quantized,
    float* scales,
    int8_t* zero_points,
    int total_size,
    int block_size
) {
    int num_blocks = (total_size + block_size - 1) / block_size;
    
    for (int b = 0; b < num_blocks; b++) {
        int start = b * block_size;
        int end = (start + block_size < total_size) ? start + block_size : total_size;
        
        // Find min and max in this block
        float min_val = weights[start];
        float max_val = weights[start];
        for (int i = start + 1; i < end; i++) {
            if (weights[i] < min_val) min_val = weights[i];
            if (weights[i] > max_val) max_val = weights[i];
        }
        
        // Compute scale and zero point for asymmetric quantization
        // Map [min_val, max_val] -> [-128, 127]
        float range = max_val - min_val;
        if (range < 1e-8f) range = 1e-8f; // Avoid division by zero
        
        scales[b] = range / 255.0f;
        zero_points[b] = (int8_t)(-128.0f - min_val / scales[b]);
        
        // Quantize values in this block
        // quantized = round(value / scale) + zero_point
        for (int i = start; i < end; i++) {
            float scaled = weights[i] / scales[b];
            int32_t qval = (int32_t)roundf(scaled) + zero_points[b];
            quantized[i] = (int8_t)(qval < -128 ? -128 : (qval > 127 ? 127 : qval));
        }
    }
}

// Helper function to dequantize block-wise quantized weights back to float (CPU version)
void dequantize_blockwise_cpu(
    const int8_t* quantized,
    const float* scales,
    const int8_t* zero_points,
    float* weights,
    int total_size,
    int block_size
) {
    // int num_blocks = (total_size + block_size - 1) / block_size;
    
    for (int i = 0; i < total_size; i++) {
        int block_idx = i / block_size;
        float scale = scales[block_idx];
        int8_t zero_point = zero_points[block_idx];
        int8_t quantized_val = quantized[i];
        
        weights[i] = scale * (float)(quantized_val - zero_point);
    }
}

// ============================================================================
// MXFP4 Quantization Implementation
// ============================================================================

// Helper function to convert float to FP4 mantissa (4 bits)
// FP4 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
// Returns 4-bit value (0-15)
__host__ __device__ inline uint8_t float_to_fp4_mantissa(float value, int shared_exp) {
    if (value == 0.0f) return 0;
    
    // Extract sign
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint8_t sign = (bits >> 31) & 0x1;
    
    // Get exponent and mantissa
    int exp = ((bits >> 23) & 0xFF) - 127; // Unbias exponent
    uint32_t mantissa = bits & 0x7FFFFF;
    
    // Adjust based on shared exponent
    int exp_diff = exp - shared_exp;
    
    // FP4: 1 sign bit + 2 exp bits + 1 mantissa bit
    // Exponent range: -2 to +1 (relative to shared exponent)
    uint8_t fp4_exp;
    if (exp_diff < -2) {
        fp4_exp = 0; // Underflow, set to smallest
        mantissa = 0;
    } else if (exp_diff > 1) {
        fp4_exp = 3; // Overflow, set to largest
        mantissa = 0x7FFFFF;
    } else {
        fp4_exp = exp_diff + 2; // Map [-2, 1] to [0, 3]
    }
    
    // Extract top mantissa bit
    uint8_t fp4_mantissa = (mantissa >> 22) & 0x1;
    
    // Combine: [sign:1][exp:2][mantissa:1]
    return (sign << 3) | (fp4_exp << 1) | fp4_mantissa;
}

// Helper function to convert FP4 mantissa back to float
__host__ __device__ inline float fp4_mantissa_to_float(uint8_t fp4_val, int shared_exp) {
    if (fp4_val == 0) return 0.0f;
    
    // Extract components
    uint8_t sign = (fp4_val >> 3) & 0x1;
    uint8_t exp_bits = (fp4_val >> 1) & 0x3;
    uint8_t mantissa_bit = fp4_val & 0x1;
    
    // Compute actual exponent
    int relative_exp = (int)exp_bits - 2; // Map [0,3] back to [-2,1]
    int actual_exp = shared_exp + relative_exp;
    
    // Build mantissa (1.mantissa_bit in binary)
    uint32_t mantissa = (1 << 23) | (mantissa_bit << 22);
    
    // Construct float
    uint32_t result_bits = (sign << 31) | ((actual_exp + 127) << 23) | (mantissa & 0x7FFFFF);
    return *reinterpret_cast<float*>(&result_bits);
}

// CPU function to quantize weights to MXFP4 format
void quantize_mxfp4(
    const float* weights,
    MXFP4Block* quantized_blocks,
    int total_size
) {
    int num_blocks = (total_size + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    
    for (int b = 0; b < num_blocks; b++) {
        int start = b * MXFP4_BLOCK_SIZE;
        int end = std::min(start + MXFP4_BLOCK_SIZE, total_size);
        
        // Find the maximum absolute value in this block
        float max_abs = 0.0f;
        for (int i = start; i < end; i++) {
            float abs_val = fabsf(weights[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        
        // Compute shared exponent
        int shared_exp = 0;
        if (max_abs > 0.0f) {
            uint32_t bits = *reinterpret_cast<uint32_t*>(&max_abs);
            shared_exp = ((bits >> 23) & 0xFF) - 127;
        }
        
        quantized_blocks[b].shared_exp = (uint8_t)(shared_exp + 127); // Store biased exponent
        
        // Quantize each value in the block
        for (int i = 0; i < MXFP4_BLOCK_SIZE; i++) {
            int idx = start + i;
            float value = (idx < total_size) ? weights[idx] : 0.0f;
            
            uint8_t fp4_val = float_to_fp4_mantissa(value, shared_exp);
            
            // Pack 2 values per byte
            if (i % 2 == 0) {
                quantized_blocks[b].data[i / 2] = fp4_val << 4;
            } else {
                quantized_blocks[b].data[i / 2] |= fp4_val;
            }
        }
    }
}

// CPU function to dequantize MXFP4 weights back to float
void dequantize_mxfp4_cpu(
    const MXFP4Block* quantized_blocks,
    float* weights,
    int total_size
) {
    int num_blocks = (total_size + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    
    for (int b = 0; b < num_blocks; b++) {
        int start = b * MXFP4_BLOCK_SIZE;
        int end = std::min(start + MXFP4_BLOCK_SIZE, total_size);
        
        int shared_exp = (int)quantized_blocks[b].shared_exp - 127; // Unbias exponent
        
        for (int i = 0; i < MXFP4_BLOCK_SIZE && (start + i) < total_size; i++) {
            // Unpack 4-bit value
            uint8_t fp4_val;
            if (i % 2 == 0) {
                fp4_val = (quantized_blocks[b].data[i / 2] >> 4) & 0xF;
            } else {
                fp4_val = quantized_blocks[b].data[i / 2] & 0xF;
            }
            
            weights[start + i] = fp4_mantissa_to_float(fp4_val, shared_exp);
        }
    }
}

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

// ============================================================================
// MXFP4 GPU Kernels
// ============================================================================

// GPU Kernel for MXFP4 Dequantization
// Dequantizes MXFP4 weights to float
// W_quantized: [num_blocks] MXFP4 blocks
// W_dequantized: [total_size] output float values
__global__ void dequantize_mxfp4_kernel(
    const MXFP4Block* W_quantized,
    float* W_dequantized,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        int block_idx = idx / MXFP4_BLOCK_SIZE;
        int local_idx = idx % MXFP4_BLOCK_SIZE;
        
        int shared_exp = (int)W_quantized[block_idx].shared_exp - 127;
        
        // Unpack 4-bit value
        uint8_t fp4_val;
        if (local_idx % 2 == 0) {
            fp4_val = (W_quantized[block_idx].data[local_idx / 2] >> 4) & 0xF;
        } else {
            fp4_val = W_quantized[block_idx].data[local_idx / 2] & 0xF;
        }
        
        W_dequantized[idx] = fp4_mantissa_to_float(fp4_val, shared_exp);
    }
}

// Kernel 0: Linear Projection (X·W + b)
// X: [batch, seq_len, d_model]
// W: [d_model, d_out]
// b: [d_out]
// Output: [batch, seq_len, d_out]
__global__ void linear_projection_kernel(
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

// Kernel 0: Tiled Linear Projection (X·W + b)
// X: [batch, seq_len, d_model]
// W: [d_model, d_out]
// b: [d_out]
// Output: [batch, seq_len, d_out]
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

// Kernel 0: Fused & Tiled Linear Projection for Q, K, V
// Computes [Q|K|V] = X·[W_Q|W_K|W_V] + [b_Q|b_K|b_V] in a single kernel
// This improves arithmetic intensity by loading X once for all three projections
// X: [batch, seq_len, d_model]
// Wq, Wk, Wv: [d_model, d_k/d_v]
// bq, bk, bv: [d_k/d_v]
// Q, K, V: [batch, seq_len, d_k/d_v]
__global__ void fused_qkv_projection_tiled_kernel(
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
/// X: [batch, seq_len, d_model]
// Wq_quantized, Wk_quantized, Wv_quantized: [d_model, d_k/d_v] int8
// scales, zero_points: per-block quantization parameters
// Q, K, V: [batch, seq_len, d_k/d_v]
__global__ void fused_dequant_qkv_projection_kernel(
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

// ============================================================================
// Fused MXFP4 Dequantization + QKV Projection Kernel
// ============================================================================

// Kernel: Fused MXFP4 Dequantization + QKV Projection
// Performs on-the-fly MXFP4 dequantization during matrix multiplication
// X: [batch, seq_len, d_model]
// Wq_quantized, Wk_quantized, Wv_quantized: MXFP4Block arrays
// Q, K, V: [batch, seq_len, d_k/d_v]
__global__ void fused_mxfp4_qkv_projection_kernel(
    const float* __restrict__ X,
    const MXFP4Block* __restrict__ Wq_quantized,
    const MXFP4Block* __restrict__ Wk_quantized,
    const MXFP4Block* __restrict__ Wv_quantized,
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
    int d_v
) {
    // Padded shared memory to avoid bank conflicts
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
        // 1. Load X tile once (shared for all three projections)
        const int x_col = t * TILE_SIZE + threadIdx.x;
        if (b_idx < batch && row < seq_len && x_col < d_model) {
            X_tile[threadIdx.y][threadIdx.x] = __ldg(&X[x_base + x_col]);
        } else {
            X_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        const int w_row = t * TILE_SIZE + threadIdx.y;
        const int w_col = col;

        // 2. Load and dequantize ALL weight tiles in parallel
        // Dequantize Wq from MXFP4
        if (w_row < d_model && w_col < d_k) {
            const int wq_idx = w_row * d_k + w_col;
            const int block_idx = wq_idx / MXFP4_BLOCK_SIZE;
            const int local_idx = wq_idx % MXFP4_BLOCK_SIZE;
            
            const int shared_exp = (int)Wq_quantized[block_idx].shared_exp - 127;
            
            // Unpack 4-bit value
            uint8_t fp4_val;
            if (local_idx % 2 == 0) {
                fp4_val = (Wq_quantized[block_idx].data[local_idx / 2] >> 4) & 0xF;
            } else {
                fp4_val = Wq_quantized[block_idx].data[local_idx / 2] & 0xF;
            }
            
            Wq_tile[threadIdx.y][threadIdx.x] = fp4_mantissa_to_float(fp4_val, shared_exp);

            // Dequantize Wk from MXFP4
            const int wk_idx = w_row * d_k + w_col;
            const int block_idx_k = wk_idx / MXFP4_BLOCK_SIZE;
            const int local_idx_k = wk_idx % MXFP4_BLOCK_SIZE;
            
            const int shared_exp_k = (int)Wk_quantized[block_idx_k].shared_exp - 127;
            
            uint8_t fp4_val_k;
            if (local_idx_k % 2 == 0) {
                fp4_val_k = (Wk_quantized[block_idx_k].data[local_idx_k / 2] >> 4) & 0xF;
            } else {
                fp4_val_k = Wk_quantized[block_idx_k].data[local_idx_k / 2] & 0xF;
            }
            
            Wk_tile[threadIdx.y][threadIdx.x] = fp4_mantissa_to_float(fp4_val_k, shared_exp_k);
        } else {
            Wq_tile[threadIdx.y][threadIdx.x] = 0.0f;
            Wk_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Dequantize Wv from MXFP4
        if (w_row < d_model && w_col < d_v) {
            const int wv_idx = w_row * d_v + w_col;
            const int block_idx_v = wv_idx / MXFP4_BLOCK_SIZE;
            const int local_idx_v = wv_idx % MXFP4_BLOCK_SIZE;
            
            const int shared_exp_v = (int)Wv_quantized[block_idx_v].shared_exp - 127;
            
            uint8_t fp4_val_v;
            if (local_idx_v % 2 == 0) {
                fp4_val_v = (Wv_quantized[block_idx_v].data[local_idx_v / 2] >> 4) & 0xF;
            } else {
                fp4_val_v = Wv_quantized[block_idx_v].data[local_idx_v / 2] & 0xF;
            }
            
            Wv_tile[threadIdx.y][threadIdx.x] = fp4_mantissa_to_float(fp4_val_v, shared_exp_v);
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


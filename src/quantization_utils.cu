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
// NF4 Quantization Implementation
// ============================================================================

// NF4 lookup table: 16 quantiles from a normal distribution N(0,1)
// These values are optimal for quantizing weights from a normal distribution
static const float NF4_QUANT_TABLE[16] = {
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

// Helper function to convert float to NF4 (4 bits)
// Maps input to closest quantile and returns index (0-15)
inline uint8_t float_to_nf4(float value, float absmax) {
    if (absmax < 1e-8f || value == 0.0f) return 7; // Map to zero
    
    // Normalize value to [-1, 1] range
    float normalized = value / absmax;
    normalized = fmaxf(-1.0f, fminf(1.0f, normalized));
    
    // Find closest quantile
    float min_dist = fabsf(normalized - NF4_QUANT_TABLE[0]);
    uint8_t best_idx = 0;
    
    for (int i = 1; i < 16; i++) {
        float dist = fabsf(normalized - NF4_QUANT_TABLE[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    
    return best_idx;
}

// Helper function to convert NF4 back to float
inline float nf4_to_float(uint8_t nf4_val, float absmax) {
    if (nf4_val >= 16) nf4_val = 15; // Clamp to valid range
    return NF4_QUANT_TABLE[nf4_val] * absmax;
}

// CPU function to quantize weights to NF4 format
void quantize_nf4(
    const float* weights,
    NF4Block* quantized_blocks,
    int total_size
) {
    int num_blocks = (total_size + NF4_BLOCK_SIZE - 1) / NF4_BLOCK_SIZE;
    
    for (int b = 0; b < num_blocks; b++) {
        int start = b * NF4_BLOCK_SIZE;
        int end = std::min(start + NF4_BLOCK_SIZE, total_size);
        
        // Find absolute maximum in this block
        float absmax = 0.0f;
        for (int i = start; i < end; i++) {
            float abs_val = fabsf(weights[i]);
            if (abs_val > absmax) absmax = abs_val;
        }
        
        quantized_blocks[b].absmax = absmax;
        
        // Quantize each value in the block
        for (int i = 0; i < NF4_BLOCK_SIZE; i++) {
            int idx = start + i;
            float value = (idx < total_size) ? weights[idx] : 0.0f;
            
            uint8_t nf4_val = float_to_nf4(value, absmax);
            
            // Pack 2 values per byte
            if (i % 2 == 0) {
                quantized_blocks[b].data[i / 2] = nf4_val << 4;
            } else {
                quantized_blocks[b].data[i / 2] |= nf4_val;
            }
        }
    }
}

// CPU function to dequantize NF4 weights back to float
void dequantize_nf4_cpu(
    const NF4Block* quantized_blocks,
    float* weights,
    int total_size
) {
    int num_blocks = (total_size + NF4_BLOCK_SIZE - 1) / NF4_BLOCK_SIZE;
    
    for (int b = 0; b < num_blocks; b++) {
        int start = b * NF4_BLOCK_SIZE;
        int end = std::min(start + NF4_BLOCK_SIZE, total_size);
        
        float absmax = quantized_blocks[b].absmax;
        
        for (int i = 0; i < NF4_BLOCK_SIZE && (start + i) < total_size; i++) {
            // Unpack 4-bit value
            uint8_t nf4_val;
            if (i % 2 == 0) {
                nf4_val = (quantized_blocks[b].data[i / 2] >> 4) & 0xF;
            } else {
                nf4_val = quantized_blocks[b].data[i / 2] & 0xF;
            }
            
            weights[start + i] = nf4_to_float(nf4_val, absmax);
        }
    }
}

// ============================================================================
// FP8 E4M3 Helper Functions (for NVFP4)
// ============================================================================

// Convert float to FP8 E4M3 format (used for NVFP4 scale factors)
// E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits, bias=7
__host__ __device__ inline uint8_t float_to_fp8_e4m3(float value) {
    if (value == 0.0f) return 0;
    
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint8_t sign = (bits >> 31) & 0x1;
    int exp = ((bits >> 23) & 0xFF) - 127;  // Unbias FP32 exponent
    uint32_t mantissa = bits & 0x7FFFFF;
    
    // E4M3 bias is 7, range is [-6, 15] (exponent values 0-15)
    // Adjust exponent for E4M3 bias
    exp += 7;
    
    // Handle overflow (exp >= 16)
    if (exp >= 15) {  // Max normal exponent for E4M3 is 15 (no infinities in E4M3)
        // Clamp to max value: S 1111 111 = ±448
        return (sign << 7) | 0x7F;
    }
    
    // Handle underflow (exp <= 0)
    if (exp <= 0) {
        // Subnormal or zero
        if (exp < -3) return (sign << 7);  // Flush to zero
        
        // Subnormal: shift mantissa right and set exp to 0
        int shift = 1 - exp;  // How many positions to shift
        mantissa = (1 << 23) | mantissa;  // Add implicit 1
        mantissa >>= (20 + shift);  // Shift to 3-bit mantissa + denorm shift
        return (sign << 7) | (mantissa & 0x7);
    }
    
    // Normal case: round mantissa from 23 bits to 3 bits
    // Round to nearest, ties to even
    uint32_t mantissa_3bit = mantissa >> 20;  // Top 3 bits
    uint32_t round_bit = (mantissa >> 19) & 0x1;
    uint32_t sticky_bits = mantissa & 0x7FFFF;
    
    if (round_bit && (sticky_bits != 0 || (mantissa_3bit & 0x1))) {
        mantissa_3bit++;
        if (mantissa_3bit > 7) {
            mantissa_3bit = 0;
            exp++;
            if (exp >= 15) {
                return (sign << 7) | 0x7F;  // Overflow to max
            }
        }
    }
    
    return (sign << 7) | (exp << 3) | (mantissa_3bit & 0x7);
}

// Convert FP8 E4M3 to float
__host__ __device__ inline float fp8_e4m3_to_float(uint8_t fp8_val) {
    uint8_t sign = (fp8_val >> 7) & 0x1;
    uint8_t exp = (fp8_val >> 3) & 0xF;
    uint8_t mantissa = fp8_val & 0x7;
    
    if (exp == 0) {
        if (mantissa == 0) {
            // Zero
            return sign ? -0.0f : 0.0f;
        }
        // Subnormal: value = (-1)^S × 2^(-6) × (0.mantissa)
        float val = mantissa / 8.0f;  // 0.mantissa in base-2
        val *= powf(2.0f, -6.0f);
        return sign ? -val : val;
    }
    
    // Normal: value = (-1)^S × 2^(E-7) × (1.mantissa)
    float val = 1.0f + mantissa / 8.0f;
    val *= powf(2.0f, (int)exp - 7);
    return sign ? -val : val;
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
    
    // OCP Spec requires roundTiesToEven (banker's rounding)
    // For 1-bit mantissa quantization, we check bit 22 (target) and bit 21 (first discarded bit)
    // Round up if: bit21=1 AND (bit22=1 OR any lower bit is 1)
    uint32_t target_bit = (mantissa >> 22) & 0x1;
    uint32_t guard_bit = (mantissa >> 21) & 0x1;
    uint32_t sticky_bits = mantissa & 0x1FFFFF; // All bits below guard bit
    
    // roundTiesToEven: round to nearest, ties to even
    if (guard_bit) {
        if (sticky_bits != 0 || target_bit == 1) {
            // Round up: either not a tie (sticky!=0) or tie to even (target=1, want to go to 0)
            mantissa += 0x400000; // Add to bit 22
            if (mantissa & 0x800000) {
                // Overflow into implicit bit, increment exponent
                mantissa = 0;
                exp++;
            }
        }
        // else: exact tie with target_bit=0 (already even), round down (do nothing)
    }
    
    // Adjust based on shared exponent
    int exp_diff = exp - shared_exp;
    
    // OCP FP4 (E2M1) Encoding per Table 5:
    // exp_bits mapping with bias=1:
    //   11 (3) -> actual_exp = 3-1 = 2  -> represents values with exp = shared_exp + 2
    //   10 (2) -> actual_exp = 2-1 = 1  -> represents values with exp = shared_exp + 1
    //   01 (1) -> actual_exp = 1-1 = 0  -> represents values with exp = shared_exp + 0
    //   00 (0) -> subnormal, exp = 0    -> represents values with exp = shared_exp - 1 (as 0.5×2^0 = 1.0×2^-1)
    
    uint8_t fp4_exp;
    uint8_t fp4_mantissa;
    
    if (exp_diff > 2) {
        // Overflow: Clamp to max normal (OCP Spec Section 5.3.3)
        fp4_exp = 3;         // 11
        fp4_mantissa = 1;    // Max: 1.1_2 × 2^2 = 6.0
    } else if (exp_diff >= 0) {
        // Normal range: map [0, 1, 2] to [01, 10, 11]
        fp4_exp = exp_diff + 1;
        fp4_mantissa = (mantissa >> 22) & 0x1;
    } else if (exp_diff == -1) {
        // Subnormal range: Convert 1.m × 2^(shared_exp-1) → 0.M × 2^shared_exp
        // Example: 1.0 × 2^-1 = 0.5 → FP4 subnormal 0.1 × 2^0, so M=1
        //          1.1 × 2^-1 = 0.75 → FP4 subnormal 0.1 × 2^0, so M=1 (rounded)
        fp4_exp = 0;  // 00
        
        // Reconstruct full mantissa (implicit 1 + fractional bits)
        // full_mantissa = 1.mantissa_bits
        // To convert to subnormal: 1.m × 2^-1 = 0.(1m) × 2^0
        // So we need to check if (1 + mantissa/2^23) * 0.5 >= 0.5
        // Simplified: always set to 1 for values in [0.5, 1.0) before scaling
        // More precisely: check if rounded value >= 0.75 (would round to 1.0)
        
        // After rounding, if the value is >= 0.75, it should round up to 1.0 (next normal)
        // But we're in exp_diff == -1, meaning post-rounding it's still < 1.0
        // So for subnormal: 0.5 ≤ val < 1.0 → M = 1
        // Values < 0.5 are already flushed to zero (exp_diff < -1)
        
        fp4_mantissa = 1;  // Subnormal in [0.5, 1.0) → 0.1 × 2^shared_exp
    } else {
        // Underflow: exp_diff < -1, flush to zero (OCP Spec Section 5.3.3)
        fp4_exp = 0;
        fp4_mantissa = 0;  // Zero: S 00 0
    }
    
    // Combine: [sign:1][exp:2][mantissa:1]
    return (sign << 3) | (fp4_exp << 1) | fp4_mantissa;
}

// Helper function to convert FP4 mantissa back to float
__host__ __device__ inline float fp4_mantissa_to_float(uint8_t fp4_val, int shared_exp) {
    // Extract components: [S:1][E:2][M:1]
    uint8_t sign = (fp4_val >> 3) & 0x1;
    uint8_t exp_bits = (fp4_val >> 1) & 0x3;
    uint8_t mantissa_bit = fp4_val & 0x1;
    
    // OCP FP4 (E2M1) Decoding per Table 5:
    // Format: val = (-1)^S × 2^(E-bias) × (1.M) for normal
    //         val = (-1)^S × 2^(1-bias) × (0.M) for subnormal (E=0)
    // bias = 1
    
    if (exp_bits == 0 && mantissa_bit == 0) {
        // Zero: S 00 0 → ±0.0
        return 0.0f;
    }
    
    int relative_exp;
    uint32_t mantissa;
    
    if (exp_bits == 0) {
        // Subnormal: S 00 1 → ±0.5 (before scaling by shared_exp)
        // Formula: (-1)^S × 2^(1-1) × (0 + 2^-1 × 1) = (-1)^S × 0.5
        // We represent 0.5 as 1.0 × 2^-1 in IEEE 754
        relative_exp = -1;
        mantissa = (1 << 23);  // 1.0 (no fractional bits)
    } else {
        // Normal: exp_bits ∈ {1,2,3}
        // E=1 (01) → 2^(1-1) = 2^0  → relative_exp = 0
        // E=2 (10) → 2^(2-1) = 2^1  → relative_exp = 1
        // E=3 (11) → 2^(3-1) = 2^2  → relative_exp = 2
        relative_exp = (int)exp_bits - 1;
        mantissa = (1 << 23) | (mantissa_bit << 22);  // 1.mantissa_bit
    }
    
    int actual_exp = shared_exp + relative_exp;
    
    // Construct IEEE 754 float
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
        
        // Compute shared exponent per OCP Spec Section 6.3:
        // X = largest power-of-two <= max(|v_i|) / largest power-of-two representable in element
        // For FP4 (E2M1), largest representable power-of-two = 2^2 = 4
        // Therefore: X = floor_pow2(max_abs) / 4 = 2^(exp_of_max - 2)
        int shared_exp = -127;
        if (max_abs > 0.0f) {
            uint32_t bits = *reinterpret_cast<uint32_t*>(&max_abs);
            int exp_of_max = ((bits >> 23) & 0xFF) - 127;
            
            // OCP-compliant: Shift down by 2 to account for FP4 max exponent of 2
            // This ensures scaled values fall within FP4 range [0.5, 6.0]
            shared_exp = exp_of_max - 2;
            
            // Clamp to valid E8M0 range [-127, 127]
            if (shared_exp < -127) shared_exp = -127;
            if (shared_exp > 127) shared_exp = 127;
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
// NF4 GPU Kernels
// ============================================================================

// GPU constant memory for NF4 lookup table
__constant__ float NF4_QUANT_TABLE_GPU[16] = {
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

// GPU constant memory for NVFP4 E2M1 lookup table
__constant__ float NVFP4_E2M1_TABLE_GPU[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// GPU Kernel for NF4 Dequantization
// Dequantizes NF4 weights to float
// W_quantized: [num_blocks] NF4 blocks
// W_dequantized: [total_size] output float values
__global__ void dequantize_nf4_kernel(
    const NF4Block* W_quantized,
    float* W_dequantized,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        int block_idx = idx / NF4_BLOCK_SIZE;
        int local_idx = idx % NF4_BLOCK_SIZE;
        
        float absmax = W_quantized[block_idx].absmax;
        
        // Unpack 4-bit value
        uint8_t nf4_val;
        if (local_idx % 2 == 0) {
            nf4_val = (W_quantized[block_idx].data[local_idx / 2] >> 4) & 0xF;
        } else {
            nf4_val = W_quantized[block_idx].data[local_idx / 2] & 0xF;
        }
        
        // Dequantize using lookup table
        if (nf4_val >= 16) nf4_val = 15;
        W_dequantized[idx] = NF4_QUANT_TABLE_GPU[nf4_val] * absmax;
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

// ============================================================================
// NVFP4 Quantization Implementation
// ============================================================================

// E2M1 lookup table for NVFP4 - CPU version
// Format: S EE M where S=sign, E=exponent(2 bits), M=mantissa(1 bit)
// Positive values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
static const float NVFP4_E2M1_TABLE_CPU[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,  // Positive (S=0)
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // Negative (S=1)
};

// Helper function to convert float to FP4 mantissa with E4M3 scaling (NVFP4 format)
// NVFP4 uses simple multiplication: x = x_q × scale, where x_q is E2M1 in range [-6, 6]
inline uint8_t float_to_nvfp4_mantissa(float value, float scale_e4m3) {
    if (scale_e4m3 < 1e-10f || value == 0.0f) return 0;
    
    // Scale the value: x_q = value / scale
    float scaled = value / scale_e4m3;
    
    // Clamp to E2M1 range [-6, 6]
    if (scaled > 6.0f) scaled = 6.0f;
    if (scaled < -6.0f) scaled = -6.0f;
    
    // Find closest E2M1 value
    uint8_t best_idx = 0;
    float min_dist = fabsf(scaled - NVFP4_E2M1_TABLE_CPU[0]);
    
    for (int i = 1; i < 16; i++) {
        float dist = fabsf(scaled - NVFP4_E2M1_TABLE_CPU[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    
    return best_idx;
}

// Helper function to convert NVFP4 mantissa back to float (CPU version)
inline float nvfp4_mantissa_to_float(uint8_t fp4_val, float scale_e4m3) {
    if (fp4_val >= 16) fp4_val = 15;  // Clamp to valid range
    
    // Direct lookup from E2M1 table
    float unscaled = NVFP4_E2M1_TABLE_CPU[fp4_val];
    
    // Apply scale: x = x_q × scale
    return unscaled * scale_e4m3;
}

// CPU function to quantize weights to NVFP4 format
void quantize_nvfp4(
    const float* weights,
    NVFP4Block* quantized_blocks,
    int total_size
) {
    int num_blocks = (total_size + NVFP4_BLOCK_SIZE - 1) / NVFP4_BLOCK_SIZE;
    
    for (int b = 0; b < num_blocks; b++) {
        int start = b * NVFP4_BLOCK_SIZE;
        int end = std::min(start + NVFP4_BLOCK_SIZE, total_size);
        
        // Find the maximum absolute value in this block
        float max_abs = 0.0f;
        for (int i = start; i < end; i++) {
            float abs_val = fabsf(weights[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        
        // Compute E4M3 scale factor
        // Target: scale max_abs to fit in FP4 range [0, 6.0]
        // Scale = max_abs / 6.0, but we want the optimal E4M3 scale
        // that minimizes quantization error across the block
        
        float target_scale = (max_abs > 1e-10f) ? (max_abs / 6.0f) : 1e-10f;
        
        // Quantize scale to E4M3 format
        uint8_t scale_e4m3_bits = float_to_fp8_e4m3(target_scale);
        quantized_blocks[b].scale_e4m3 = scale_e4m3_bits;
        
        // Dequantize to get actual scale
        float actual_scale = fp8_e4m3_to_float(scale_e4m3_bits);
        
        // Quantize each value in the block
        for (int i = 0; i < NVFP4_BLOCK_SIZE; i++) {
            int idx = start + i;
            float value = (idx < total_size) ? weights[idx] : 0.0f;
            
            uint8_t nvfp4_val = float_to_nvfp4_mantissa(value, actual_scale);
            
            // Pack 2 values per byte
            if (i % 2 == 0) {
                quantized_blocks[b].data[i / 2] = nvfp4_val << 4;
            } else {
                quantized_blocks[b].data[i / 2] |= nvfp4_val;
            }
        }
    }
}

// CPU function to dequantize NVFP4 weights back to float
void dequantize_nvfp4_cpu(
    const NVFP4Block* quantized_blocks,
    float* weights,
    int total_size
) {
    int num_blocks = (total_size + NVFP4_BLOCK_SIZE - 1) / NVFP4_BLOCK_SIZE;
    
    for (int b = 0; b < num_blocks; b++) {
        int start = b * NVFP4_BLOCK_SIZE;
        int end = std::min(start + NVFP4_BLOCK_SIZE, total_size);
        
        // Dequantize E4M3 scale
        float scale_e4m3 = fp8_e4m3_to_float(quantized_blocks[b].scale_e4m3);
        
        for (int i = 0; i < NVFP4_BLOCK_SIZE && (start + i) < total_size; i++) {
            // Unpack 4-bit value
            uint8_t nvfp4_val;
            if (i % 2 == 0) {
                nvfp4_val = (quantized_blocks[b].data[i / 2] >> 4) & 0xF;
            } else {
                nvfp4_val = quantized_blocks[b].data[i / 2] & 0xF;
            }
            
            weights[start + i] = nvfp4_mantissa_to_float(nvfp4_val, scale_e4m3);
        }
    }
}

// ============================================================================
// NVFP4 GPU Kernels
// ============================================================================

// GPU Kernel for NVFP4 Dequantization
// Dequantizes NVFP4 weights to float
// W_quantized: [num_blocks] NVFP4 blocks
// W_dequantized: [total_size] output float values
__global__ void dequantize_nvfp4_kernel(
    const NVFP4Block* W_quantized,
    float* W_dequantized,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        int block_idx = idx / NVFP4_BLOCK_SIZE;
        int local_idx = idx % NVFP4_BLOCK_SIZE;
        
        float scale_e4m3 = fp8_e4m3_to_float(W_quantized[block_idx].scale_e4m3);
        
        // Unpack 4-bit value
        uint8_t nvfp4_val;
        if (local_idx % 2 == 0) {
            nvfp4_val = (W_quantized[block_idx].data[local_idx / 2] >> 4) & 0xF;
        } else {
            nvfp4_val = W_quantized[block_idx].data[local_idx / 2] & 0xF;
        }
        
        // Dequantize using lookup table: x = x_q × scale
        if (nvfp4_val >= 16) nvfp4_val = 15;
        W_dequantized[idx] = NVFP4_E2M1_TABLE_GPU[nvfp4_val] * scale_e4m3;
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
// Fused NF4 Dequantization + QKV Projection Kernel
// ============================================================================

// Kernel: Fused NF4 Dequantization + QKV Projection
// Performs on-the-fly NF4 dequantization during matrix multiplication
// X: [batch, seq_len, d_model]
// Wq_quantized, Wk_quantized, Wv_quantized: NF4Block arrays
// Q, K, V: [batch, seq_len, d_k/d_v]
__global__ void fused_nf4_qkv_projection_kernel(
    const float* __restrict__ X,
    const NF4Block* __restrict__ Wq_quantized,
    const NF4Block* __restrict__ Wk_quantized,
    const NF4Block* __restrict__ Wv_quantized,
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
        // Dequantize Wq from NF4
        if (w_row < d_model && w_col < d_k) {
            const int wq_idx = w_row * d_k + w_col;
            const int block_idx = wq_idx / NF4_BLOCK_SIZE;
            const int local_idx = wq_idx % NF4_BLOCK_SIZE;
            
            const float absmax = Wq_quantized[block_idx].absmax;
            
            // Unpack 4-bit value
            uint8_t nf4_val;
            if (local_idx % 2 == 0) {
                nf4_val = (Wq_quantized[block_idx].data[local_idx / 2] >> 4) & 0xF;
            } else {
                nf4_val = Wq_quantized[block_idx].data[local_idx / 2] & 0xF;
            }
            
            if (nf4_val >= 16) nf4_val = 15;
            Wq_tile[threadIdx.y][threadIdx.x] = NF4_QUANT_TABLE_GPU[nf4_val] * absmax;

            // Dequantize Wk from NF4
            const int wk_idx = w_row * d_k + w_col;
            const int block_idx_k = wk_idx / NF4_BLOCK_SIZE;
            const int local_idx_k = wk_idx % NF4_BLOCK_SIZE;
            
            const float absmax_k = Wk_quantized[block_idx_k].absmax;
            
            uint8_t nf4_val_k;
            if (local_idx_k % 2 == 0) {
                nf4_val_k = (Wk_quantized[block_idx_k].data[local_idx_k / 2] >> 4) & 0xF;
            } else {
                nf4_val_k = Wk_quantized[block_idx_k].data[local_idx_k / 2] & 0xF;
            }
            
            if (nf4_val_k >= 16) nf4_val_k = 15;
            Wk_tile[threadIdx.y][threadIdx.x] = NF4_QUANT_TABLE_GPU[nf4_val_k] * absmax_k;
        } else {
            Wq_tile[threadIdx.y][threadIdx.x] = 0.0f;
            Wk_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Dequantize Wv from NF4
        if (w_row < d_model && w_col < d_v) {
            const int wv_idx = w_row * d_v + w_col;
            const int block_idx_v = wv_idx / NF4_BLOCK_SIZE;
            const int local_idx_v = wv_idx % NF4_BLOCK_SIZE;
            
            const float absmax_v = Wv_quantized[block_idx_v].absmax;
            
            uint8_t nf4_val_v;
            if (local_idx_v % 2 == 0) {
                nf4_val_v = (Wv_quantized[block_idx_v].data[local_idx_v / 2] >> 4) & 0xF;
            } else {
                nf4_val_v = Wv_quantized[block_idx_v].data[local_idx_v / 2] & 0xF;
            }
            
            if (nf4_val_v >= 16) nf4_val_v = 15;
            Wv_tile[threadIdx.y][threadIdx.x] = NF4_QUANT_TABLE_GPU[nf4_val_v] * absmax_v;
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

// ============================================================================
// Fused NVFP4 Dequantization + QKV Projection Kernel
// ============================================================================

// Kernel: Fused NVFP4 Dequantization + QKV Projection
// Performs on-the-fly NVFP4 dequantization during matrix multiplication
// X: [batch, seq_len, d_model]
// Wq_quantized, Wk_quantized, Wv_quantized: NVFP4Block arrays
// Q, K, V: [batch, seq_len, d_k/d_v]
__global__ void fused_nvfp4_qkv_projection_kernel(
    const float* __restrict__ X,
    const NVFP4Block* __restrict__ Wq_quantized,
    const NVFP4Block* __restrict__ Wk_quantized,
    const NVFP4Block* __restrict__ Wv_quantized,
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
        // Dequantize Wq from NVFP4
        if (w_row < d_model && w_col < d_k) {
            const int wq_idx = w_row * d_k + w_col;
            const int block_idx = wq_idx / NVFP4_BLOCK_SIZE;
            const int local_idx = wq_idx % NVFP4_BLOCK_SIZE;
            
            const float scale_e4m3 = fp8_e4m3_to_float(Wq_quantized[block_idx].scale_e4m3);
            
            // Unpack 4-bit value
            uint8_t nvfp4_val;
            if (local_idx % 2 == 0) {
                nvfp4_val = (Wq_quantized[block_idx].data[local_idx / 2] >> 4) & 0xF;
            } else {
                nvfp4_val = Wq_quantized[block_idx].data[local_idx / 2] & 0xF;
            }
            
            if (nvfp4_val >= 16) nvfp4_val = 15;
            Wq_tile[threadIdx.y][threadIdx.x] = NVFP4_E2M1_TABLE_GPU[nvfp4_val] * scale_e4m3;

            // Dequantize Wk from NVFP4
            const int wk_idx = w_row * d_k + w_col;
            const int block_idx_k = wk_idx / NVFP4_BLOCK_SIZE;
            const int local_idx_k = wk_idx % NVFP4_BLOCK_SIZE;
            
            const float scale_e4m3_k = fp8_e4m3_to_float(Wk_quantized[block_idx_k].scale_e4m3);
            
            uint8_t nvfp4_val_k;
            if (local_idx_k % 2 == 0) {
                nvfp4_val_k = (Wk_quantized[block_idx_k].data[local_idx_k / 2] >> 4) & 0xF;
            } else {
                nvfp4_val_k = Wk_quantized[block_idx_k].data[local_idx_k / 2] & 0xF;
            }
            
            if (nvfp4_val_k >= 16) nvfp4_val_k = 15;
            Wk_tile[threadIdx.y][threadIdx.x] = NVFP4_E2M1_TABLE_GPU[nvfp4_val_k] * scale_e4m3_k;
        } else {
            Wq_tile[threadIdx.y][threadIdx.x] = 0.0f;
            Wk_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Dequantize Wv from NVFP4
        if (w_row < d_model && w_col < d_v) {
            const int wv_idx = w_row * d_v + w_col;
            const int block_idx_v = wv_idx / NVFP4_BLOCK_SIZE;
            const int local_idx_v = wv_idx % NVFP4_BLOCK_SIZE;
            
            const float scale_e4m3_v = fp8_e4m3_to_float(Wv_quantized[block_idx_v].scale_e4m3);
            
            uint8_t nvfp4_val_v;
            if (local_idx_v % 2 == 0) {
                nvfp4_val_v = (Wv_quantized[block_idx_v].data[local_idx_v / 2] >> 4) & 0xF;
            } else {
                nvfp4_val_v = Wv_quantized[block_idx_v].data[local_idx_v / 2] & 0xF;
            }
            
            if (nvfp4_val_v >= 16) nvfp4_val_v = 15;
            Wv_tile[threadIdx.y][threadIdx.x] = NVFP4_E2M1_TABLE_GPU[nvfp4_val_v] * scale_e4m3_v;
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

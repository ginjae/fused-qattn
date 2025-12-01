#ifndef QUANTIZATION_UTILS_CUH
#define QUANTIZATION_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_SIZE 16    // Tile size for fused_dequant_qkv_projection_kernel
#define MXFP4_BLOCK_SIZE 32  // Block size for MXFP4 quantization
#define NVFP4_BLOCK_SIZE 16  // Block size for NVFP4 quantization
#define NF4_BLOCK_SIZE 64    // Block size for NF4 quantization

// MXFP4 quantized data structure
// Each block shares one 8-bit exponent and contains MXFP4_BLOCK_SIZE 4-bit mantissas
struct MXFP4Block {
    uint8_t shared_exp;     // Shared exponent for the block
    uint8_t data[16];       // Packed 4-bit values (2 values per byte, for 32 values)
};

// NVFP4 quantized data structure (NVIDIA Blackwell format)
// Uses 2-level scaling: FP32 per-tensor + E4M3 per-block (16 elements)
// Based on arxiv:2509.25149 specification
struct NVFP4Block {
    uint8_t scale_e4m3;     // FP8 E4M3 local decode scale for the block
    uint8_t data[8];        // Packed 4-bit values (2 values per byte, for 16 values)
};

// NVFP4 Tensor metadata (stores per-tensor scale)
struct NVFP4TensorMeta {
    float global_scale_dec; // FP32 global decode scale (inverse of encode scale)
};

// NF4 quantized data structure
// Each block has one absmax scale and contains NF4_BLOCK_SIZE 4-bit values
// NF4 uses quantiles from a normal distribution for optimal quantization
struct NF4Block {
    float absmax;           // Absolute maximum value in the block
    uint8_t data[32];       // Packed 4-bit values (2 values per byte, for 64 values)
};

// Helper function to simulate block-wise quantization
void quantize_blockwise(
    const float* weights,
    int8_t* quantized,
    float* scales,
    int8_t* zero_points,
    int total_size,
    int block_size
);

// Helper function to dequantize block-wise quantized weights back to float (CPU version)
void dequantize_blockwise_cpu(
    const int8_t* quantized,
    const float* scales,
    const int8_t* zero_points,
    float* weights,
    int total_size,
    int block_size
);

// MXFP4 Quantization Functions
// Quantize weights to MXFP4 format (CPU version)
void quantize_mxfp4(
    const float* weights,
    MXFP4Block* quantized_blocks,
    int total_size
);

// Dequantize MXFP4 weights back to float (CPU version)
void dequantize_mxfp4_cpu(
    const MXFP4Block* quantized_blocks,
    float* weights,
    int total_size
);

// NVFP4 Quantization Functions
// Quantize weights to NVFP4 format (CPU version)
// Returns per-tensor metadata in meta parameter
void quantize_nvfp4(
    const float* weights,
    NVFP4Block* quantized_blocks,
    NVFP4TensorMeta* meta,
    int total_size
);

// Dequantize NVFP4 weights back to float (CPU version)
void dequantize_nvfp4_cpu(
    const NVFP4Block* quantized_blocks,
    const NVFP4TensorMeta* meta,
    float* weights,
    int total_size
);

__global__ void dequantize_blockwise_kernel(
    const int8_t* W_quantized,
    const float* scales,
    const int8_t* zero_points,
    float* W_dequantized,
    int rows,
    int cols,
    int block_size
);

// MXFP4 GPU dequantization kernel
__global__ void dequantize_mxfp4_kernel(
    const MXFP4Block* W_quantized,
    float* W_dequantized,
    int total_size
);

// NVFP4 GPU dequantization kernel
__global__ void dequantize_nvfp4_kernel(
    const NVFP4Block* W_quantized,
    float s_dec_global,
    float* W_dequantized,
    int total_size
);

// NF4 Quantization Functions
// Quantize weights to NF4 format (CPU version)
void quantize_nf4(
    const float* weights,
    NF4Block* quantized_blocks,
    int total_size
);

// Dequantize NF4 weights back to float (CPU version)
void dequantize_nf4_cpu(
    const NF4Block* quantized_blocks,
    float* weights,
    int total_size
);

// NF4 GPU dequantization kernel
__global__ void dequantize_nf4_kernel(
    const NF4Block* W_quantized,
    float* W_dequantized,
    int total_size
);

__global__ void linear_projection_kernel(
    const float* X,
    const float* W,
    const float* b,
    float* output,
    int batch,
    int seq_len,
    int d_model,
    int d_out
);

__global__ void linear_projection_tiled_kernel(
    const float* X,
    const float* W,
    const float* b,
    float* output,
    int batch,
    int seq_len,
    int d_model,
    int d_out
);

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
);

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
);

// Fused MXFP4 Dequantization + QKV Projection Kernel
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
);

__global__ void fused_mxfp4_qkv_dequant_kernel(
    const MXFP4Block* __restrict__ Wq_quantized,
    const MXFP4Block* __restrict__ Wk_quantized,
    const MXFP4Block* __restrict__ Wv_quantized,
    float* __restrict__ Wq_out,
    float* __restrict__ Wk_out,
    float* __restrict__ Wv_out,
    int d_model,
    int d_k,
    int d_v
);

// Fused NF4 Dequantization + QKV Projection Kernel
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
);

__global__ void fused_nf4_qkv_dequant_kernel(
    const NF4Block* __restrict__ Wq_quantized,
    const NF4Block* __restrict__ Wk_quantized,
    const NF4Block* __restrict__ Wv_quantized,
    float* __restrict__ Wq_out,
    float* __restrict__ Wk_out,
    float* __restrict__ Wv_out,
    int d_model,
    int d_k,
    int d_v
);

// Fused NVFP4 Dequantization + QKV Projection Kernel
__global__ void fused_nvfp4_qkv_projection_kernel(
    const float* __restrict__ X,
    const NVFP4Block* __restrict__ Wq_quantized,
    const NVFP4Block* __restrict__ Wk_quantized,
    const NVFP4Block* __restrict__ Wv_quantized,
    float s_dec_global_q,
    float s_dec_global_k,
    float s_dec_global_v,
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
);

__global__ void fused_nvfp4_qkv_dequant_kernel(
    const NVFP4Block* __restrict__ Wq_quantized,
    const NVFP4Block* __restrict__ Wk_quantized,
    const NVFP4Block* __restrict__ Wv_quantized,
    float s_dec_global_q,
    float s_dec_global_k,
    float s_dec_global_v,
    float* __restrict__ Wq_dequant,
    float* __restrict__ Wk_dequant,
    float* __restrict__ Wv_dequant,
    int d_model,
    int d_k,
    int d_v
);

#endif // QUANTIZATION_UTILS_CUH

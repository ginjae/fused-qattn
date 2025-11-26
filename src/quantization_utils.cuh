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

// MXFP4 quantized data structure
// Each block shares one 8-bit exponent and contains MXFP4_BLOCK_SIZE 4-bit mantissas
struct MXFP4Block {
    uint8_t shared_exp;     // Shared exponent for the block
    uint8_t data[16];       // Packed 4-bit values (2 values per byte, for 32 values)
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

#endif // QUANTIZATION_UTILS_CUH

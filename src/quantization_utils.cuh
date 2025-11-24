#ifndef QUANTIZATION_UTILS_CUH
#define QUANTIZATION_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

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

#endif // QUANTIZATION_UTILS_CUH

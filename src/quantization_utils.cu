#include "quantization_utils.cuh"
#include <algorithm>

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

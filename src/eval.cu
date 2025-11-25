#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "quantization_utils.cuh"
#include "attn_naive.cuh"
#include "attn_tiled.cuh"
#include "attn_flash.cuh"
#include "attn_ours.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void print_output_sample(const char* label, float* output, int seq_len, int d_v) {
    printf("%s - showing first 3 and last 3 positions:\n", label);
    for (int i = 0; i < seq_len; i++) {
        if (i < 3 || i >= seq_len - 3) {
            printf("Position %d: [", i);
            for (int j = 0; j < (d_v < 8 ? d_v : 8); j++) {
                int idx = i * d_v + j;
                printf("%.6f", output[idx]);
                if (j < (d_v < 8 ? d_v : 8) - 1) printf(", ");
            }
            if (d_v > 8) printf(", ...");
            printf("]\n");
        } else if (i == 3) {
            printf("... (%d more positions) ...\n", seq_len - 6);
        }
    }
}

// Function to compute error metrics between two outputs
void compute_error_metrics(const char* label, float* output_ref, float* output_test, 
                          int batch, int seq_len, int d_v) {
    int total_elements = batch * seq_len * d_v;
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    double sum_squared_error = 0.0;
    double sum_ref_squared = 0.0;
    
    for (int i = 0; i < total_elements; i++) {
        double diff = fabs(output_test[i] - output_ref[i]);
        max_abs_error = fmax(max_abs_error, diff);
        sum_abs_error += diff;
        sum_squared_error += diff * diff;
        sum_ref_squared += output_ref[i] * output_ref[i];
    }
    
    double mean_abs_error = sum_abs_error / total_elements;
    double rmse = sqrt(sum_squared_error / total_elements);
    double relative_error = sqrt(sum_squared_error / sum_ref_squared);
    
    printf("  %s Correctness Check:\n", label);
    printf("    Max Absolute Error: %.6e\n", max_abs_error);
    printf("    Mean Absolute Error: %.6e\n", mean_abs_error);
    printf("    RMSE: %.6e\n", rmse);
    printf("    Relative Error: %.6e\n", relative_error);
    
    // Determine if the result is acceptable (you can adjust threshold)
    if (max_abs_error < 1e-3 && relative_error < 1e-3) {
        printf("    Status: ✓ PASS (High precision match)\n");
    } else if (max_abs_error < 1e-2 && relative_error < 1e-2) {
        printf("    Status: ✓ PASS (Acceptable precision)\n");
    } else if (max_abs_error < 1e-1 && relative_error < 1e-1) {
        printf("    Status: ⚠ WARNING (Lower precision)\n");
    } else {
        printf("    Status: ✗ FAIL (Significant error)\n");
    }
}


int main() {
    // Test parameters (GPT-2 small scale)
    int batch = 1;
    int seq_len = 128;
    int d_model = 768;
    int d_k = 64;
    int d_v = 64;
    int block_size = 64; // Block size for quantization

    printf("\n=== Starting Evaluation (GPT-2 Scale, Unquantized weights) ===\n");

    printf("\n== Initializing Weights ==\n");
    // Allocate and initialize host memory for input and weights
    size_t x_size = batch * seq_len * d_model * sizeof(float);
    size_t wq_size = d_model * d_k * sizeof(float);
    size_t wk_size = d_model * d_k * sizeof(float);
    size_t wv_size = d_model * d_v * sizeof(float);
    size_t bq_size = d_k * sizeof(float);
    size_t bk_size = d_k * sizeof(float);
    size_t bv_size = d_v * sizeof(float);
    size_t out_size = batch * seq_len * d_v * sizeof(float);

    float* h_X = (float*)malloc(x_size);
    float* h_Wq = (float*)malloc(wq_size);
    float* h_Wk = (float*)malloc(wk_size);
    float* h_Wv = (float*)malloc(wv_size);
    float* h_bq = (float*)malloc(bq_size);
    float* h_bk = (float*)malloc(bk_size);
    float* h_bv = (float*)malloc(bv_size);
    float* h_output = (float*)malloc(out_size);

    srand(42);  // For reproducibility

    // Initialize input X
    for (int i = 0; i < batch * seq_len * d_model; i++) {
        h_X[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Range: [-1, 1]
    }

    // Initialize weight matrices with deterministic random values
    for (int i = 0; i < d_model * d_k; i++) {
        h_Wq[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;    // Range: [-0.05, 0.05]
        h_Wk[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
    for (int i = 0; i < d_model * d_v; i++) {
        h_Wv[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }

    // Initialize biases with deterministic random values
    for (int i = 0; i < d_k; i++) {
        h_bq[i] = ((float)rand() / RAND_MAX) * 0.01f - 0.005f;  // Range: [-0.005, 0.005]
        h_bk[i] = ((float)rand() / RAND_MAX) * 0.01f - 0.005f;
    }
    for (int i = 0; i < d_v; i++) {
        h_bv[i] = ((float)rand() / RAND_MAX) * 0.01f - 0.005f;
    }

    // Allocate device memory
    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_bq, *d_bk, *d_bv, *d_output;
    CUDA_CHECK(cudaMalloc(&d_X, x_size));
    CUDA_CHECK(cudaMalloc(&d_Wq, wq_size));
    CUDA_CHECK(cudaMalloc(&d_Wk, wk_size));
    CUDA_CHECK(cudaMalloc(&d_Wv, wv_size));
    CUDA_CHECK(cudaMalloc(&d_bq, bq_size));
    CUDA_CHECK(cudaMalloc(&d_bk, bk_size));
    CUDA_CHECK(cudaMalloc(&d_bv, bv_size));
    CUDA_CHECK(cudaMalloc(&d_output, out_size));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, x_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wq, h_Wq, wq_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk, h_Wk, wk_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv, h_Wv, wv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bq, h_bq, bq_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bk, h_bk, bk_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bv, h_bv, bv_size, cudaMemcpyHostToDevice));

    // Allocate output buffers for both tests
    float* h_output_naive_baseline = (float*)malloc(out_size);  // Baseline for comparison
    float* h_output_unquant = (float*)malloc(out_size);
    float* h_output_quant = (float*)malloc(out_size);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed_time_unquant, elapsed_time_quant;

    // Test 1: Naive attention with original (unquantized) weights - BASELINE
    printf("\n1. Naive Attention (BASELINE)\n");

    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    naive_attention(d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv, 
                    d_output, batch, seq_len, d_model, d_k, d_v, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    naive_attention(d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv, 
                    d_output, batch, seq_len, d_model, d_k, d_v, false);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_unquant, start, stop));

    // Copy result back as baseline
    CUDA_CHECK(cudaMemcpy(h_output_naive_baseline, d_output, out_size, cudaMemcpyDeviceToHost));

    // Print results (first 3 and last 3 positions only)
    // print_output_sample("Output (Unquantized)", h_output_naive_baseline, seq_len, d_v);
    printf("Execution time: %.4f ms\n", elapsed_time_unquant);


    // Test 2: Tiled attention with original (unquantized) weights
    printf("\n2. Tiled Attention\n");

    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    tiled_attention(d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv, 
                    d_output, batch, seq_len, d_model, d_k, d_v, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    tiled_attention(d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv, 
                    d_output, batch, seq_len, d_model, d_k, d_v, false);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_unquant, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output_unquant, d_output, out_size, cudaMemcpyDeviceToHost));

    // Print results (first 3 and last 3 positions only)
    // print_output_sample("Output (Unquantized)", h_output_unquant, seq_len, d_v);
    printf("Execution time: %.4f ms\n", elapsed_time_unquant);
    
    // Compare with baseline
    compute_error_metrics("Tiled vs Naive", h_output_naive_baseline, h_output_unquant, 
                         batch, seq_len, d_v);


    // Test 3: Flash-style attention with original (unquantized) weights
    printf("\n3. Flash-style Attention\n");

    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    flash_attention(d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv, 
                    d_output, batch, seq_len, d_model, d_k, d_v, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    flash_attention(d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv, 
                    d_output, batch, seq_len, d_model, d_k, d_v, false);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_unquant, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output_unquant, d_output, out_size, cudaMemcpyDeviceToHost));

    // Print results (first 3 and last 3 positions only)
    // print_output_sample("Output (Unquantized)", h_output_unquant, seq_len, d_v);
    printf("Execution time: %.4f ms\n", elapsed_time_unquant);
    
    // Compare with baseline
    compute_error_metrics("Flash vs Naive", h_output_naive_baseline, h_output_unquant, 
                         batch, seq_len, d_v);
    
    printf("=== End of Test ===\n");

    printf("\n");

    printf("\n=== Starting Evaluation (GPT-2 Scale, Quantized weights) ===\n");

    // Quantize weights
    printf("\n== Quantizing Weights ==\n");
    int num_blocks_q = (d_model * d_k + block_size - 1) / block_size;
    int num_blocks_v = (d_model * d_v + block_size - 1) / block_size;

    int8_t* h_Wq_quant = (int8_t*)malloc(d_model * d_k * sizeof(int8_t));
    int8_t* h_Wk_quant = (int8_t*)malloc(d_model * d_k * sizeof(int8_t));
    int8_t* h_Wv_quant = (int8_t*)malloc(d_model * d_v * sizeof(int8_t));

    float* h_Wq_scales = (float*)malloc(num_blocks_q * sizeof(float));
    float* h_Wk_scales = (float*)malloc(num_blocks_q * sizeof(float));
    float* h_Wv_scales = (float*)malloc(num_blocks_v * sizeof(float));

    int8_t* h_Wq_zeros = (int8_t*)malloc(num_blocks_q * sizeof(int8_t));
    int8_t* h_Wk_zeros = (int8_t*)malloc(num_blocks_q * sizeof(int8_t));
    int8_t* h_Wv_zeros = (int8_t*)malloc(num_blocks_v * sizeof(int8_t));

    quantize_blockwise(h_Wq, h_Wq_quant, h_Wq_scales, h_Wq_zeros, d_model * d_k, block_size);
    quantize_blockwise(h_Wk, h_Wk_quant, h_Wk_scales, h_Wk_zeros, d_model * d_k, block_size);
    quantize_blockwise(h_Wv, h_Wv_quant, h_Wv_scales, h_Wv_zeros, d_model * d_v, block_size);

    printf("Block size: %d\n", block_size);
    printf("Num blocks (Q/K): %d, Num blocks (V): %d\n", num_blocks_q, num_blocks_v);
    printf("Quantization complete.\n");

    // Allocate device memory for quantized weights
    int8_t *d_Wq_quant, *d_Wk_quant, *d_Wv_quant;
    float *d_Wq_scales, *d_Wk_scales, *d_Wv_scales;
    int8_t *d_Wq_zeros, *d_Wk_zeros, *d_Wv_zeros;

    CUDA_CHECK(cudaMalloc(&d_Wq_quant, d_model * d_k * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Wk_quant, d_model * d_k * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Wv_quant, d_model * d_v * sizeof(int8_t)));

    CUDA_CHECK(cudaMalloc(&d_Wq_scales, num_blocks_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk_scales, num_blocks_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv_scales, num_blocks_v * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Wq_zeros, num_blocks_q * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Wk_zeros, num_blocks_q * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Wv_zeros, num_blocks_v * sizeof(int8_t)));

    // Copy quantized data to device
    CUDA_CHECK(cudaMemcpy(d_Wq_quant, h_Wq_quant, d_model * d_k * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk_quant, h_Wk_quant, d_model * d_k * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv_quant, h_Wv_quant, d_model * d_v * sizeof(int8_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_Wq_scales, h_Wq_scales, num_blocks_q * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk_scales, h_Wk_scales, num_blocks_q * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv_scales, h_Wv_scales, num_blocks_v * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_Wq_zeros, h_Wq_zeros, num_blocks_q * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk_zeros, h_Wk_zeros, num_blocks_q * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv_zeros, h_Wv_zeros, num_blocks_v * sizeof(int8_t), cudaMemcpyHostToDevice));


    printf("\n1. Naive Attention (Quantized)\n");
    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    naive_attention_quantized(d_X, 
                              d_Wq_quant, d_Wk_quant, d_Wv_quant,
                              d_Wq_scales, d_Wk_scales, d_Wv_scales,
                              d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              block_size, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    // Run quantized attention
    naive_attention_quantized(d_X, 
                              d_Wq_quant, d_Wk_quant, d_Wv_quant,
                              d_Wq_scales, d_Wk_scales, d_Wv_scales,
                              d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              block_size, false);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_quant, start, stop));

    // Copy quantized result back
    CUDA_CHECK(cudaMemcpy(h_output_quant, d_output, out_size, cudaMemcpyDeviceToHost));

    // Print quantized results (first 3 and last 3 positions only)
    // print_output_sample("Output (Quantized)", h_output_quant, seq_len, d_v);
    printf("Execution time: %.4f ms\n", elapsed_time_quant);
    
    // Compare with baseline
    compute_error_metrics("Naive Quantized vs Naive Baseline", h_output_naive_baseline, h_output_quant, 
                         batch, seq_len, d_v);


    printf("\n2. Tiled Attention (Quantized)\n");
    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    tiled_attention_quantized(d_X, 
                              d_Wq_quant, d_Wk_quant, d_Wv_quant,
                              d_Wq_scales, d_Wk_scales, d_Wv_scales,
                              d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              block_size, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    // Run quantized attention
    tiled_attention_quantized(d_X, 
                              d_Wq_quant, d_Wk_quant, d_Wv_quant,
                              d_Wq_scales, d_Wk_scales, d_Wv_scales,
                              d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              block_size, false);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_quant, start, stop));

    // Copy quantized result back
    CUDA_CHECK(cudaMemcpy(h_output_quant, d_output, out_size, cudaMemcpyDeviceToHost));

    // Print quantized results (first 3 and last 3 positions only)
    // print_output_sample("Output (Quantized)", h_output_quant, seq_len, d_v);
    printf("Execution time: %.4f ms\n", elapsed_time_quant);
    
    // Compare with baseline
    compute_error_metrics("Tiled Quantized vs Naive Baseline", h_output_naive_baseline, h_output_quant, 
                         batch, seq_len, d_v);


    printf("\n3. Flash-style Attention (Quantized)\n");
    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    flash_attention_quantized(d_X, 
                              d_Wq_quant, d_Wk_quant, d_Wv_quant,
                              d_Wq_scales, d_Wk_scales, d_Wv_scales,
                              d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              block_size, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    // Run quantized attention
    flash_attention_quantized(d_X, 
                              d_Wq_quant, d_Wk_quant, d_Wv_quant,
                              d_Wq_scales, d_Wk_scales, d_Wv_scales,
                              d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              block_size, false);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_quant, start, stop));

    // Copy quantized result back
    CUDA_CHECK(cudaMemcpy(h_output_quant, d_output, out_size, cudaMemcpyDeviceToHost));

    // Print quantized results (first 3 and last 3 positions only)
    // print_output_sample("Output (Quantized)", h_output_quant, seq_len, d_v);
    printf("Execution time: %.4f ms\n", elapsed_time_quant);
    
    // Compare with baseline
    compute_error_metrics("Flash Quantized vs Naive Baseline", h_output_naive_baseline, h_output_quant, 
                         batch, seq_len, d_v);


    printf("\n4. Our Attention (Quantized)\n");
    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    our_attention_quantized(d_X, 
                              d_Wq_quant, d_Wk_quant, d_Wv_quant,
                              d_Wq_scales, d_Wk_scales, d_Wv_scales,
                              d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              block_size, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    // Run quantized attention
    our_attention_quantized(d_X, 
                              d_Wq_quant, d_Wk_quant, d_Wv_quant,
                              d_Wq_scales, d_Wk_scales, d_Wv_scales,
                              d_Wq_zeros, d_Wk_zeros, d_Wv_zeros,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              block_size, false);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_quant, start, stop));

    // Copy quantized result back
    CUDA_CHECK(cudaMemcpy(h_output_quant, d_output, out_size, cudaMemcpyDeviceToHost));

    // Print quantized results (first 3 and last 3 positions only)
    // print_output_sample("Output (Quantized)", h_output_quant, seq_len, d_v);
    printf("Execution time: %.4f ms\n", elapsed_time_quant);
    
    // Compare with baseline
    compute_error_metrics("Our Quantized vs Naive Baseline", h_output_naive_baseline, h_output_quant, 
                         batch, seq_len, d_v);


    // Cleanup CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Cleanup
    free(h_X);
    free(h_Wq);
    free(h_Wk);
    free(h_Wv);
    free(h_bq);
    free(h_bk);
    free(h_bv);
    free(h_output);
    free(h_output_naive_baseline);
    free(h_output_unquant);
    free(h_output_quant);
    free(h_Wq_quant);
    free(h_Wk_quant);
    free(h_Wv_quant);
    free(h_Wq_scales);
    free(h_Wk_scales);
    free(h_Wv_scales);
    free(h_Wq_zeros);
    free(h_Wk_zeros);
    free(h_Wv_zeros);

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaFree(d_bq));
    CUDA_CHECK(cudaFree(d_bk));
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_Wq_quant));
    CUDA_CHECK(cudaFree(d_Wk_quant));
    CUDA_CHECK(cudaFree(d_Wv_quant));
    CUDA_CHECK(cudaFree(d_Wq_scales));
    CUDA_CHECK(cudaFree(d_Wk_scales));
    CUDA_CHECK(cudaFree(d_Wv_scales));
    CUDA_CHECK(cudaFree(d_Wq_zeros));
    CUDA_CHECK(cudaFree(d_Wk_zeros));
    CUDA_CHECK(cudaFree(d_Wv_zeros));

    printf("=== End of Test ===\n");

    return 0;
}

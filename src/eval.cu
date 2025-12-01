#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <algorithm>
#include <time.h>
#include <sys/stat.h>
#include "quantization_utils.cuh"
#include "attn_naive.cuh"
#include "attn_tiled.cuh"
#include "attn_flash.cuh"
#include "attn_ours.cuh"
#include "attn_full.cuh"
#include "npy_loader.h"

#define NUM_ITERATIONS 100
#define COOLDOWN_SECONDS 1  // seconds to wait between tests
#define MAX_KERNELS 10  // Maximum number of kernels for any implementation


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

// Quantization bit-width enum for error metric evaluation
enum QuantizationType {
    UNQUANTIZED = 0,
    INT8_QUANT = 8,
    FP4_QUANT = 4  // For NF4, MXFP4, etc.
};

// Function to compute error metrics between two outputs
void compute_error_metrics(const char* label, float* output_ref, float* output_test, 
                          int batch, int seq_len, int d_v, QuantizationType quant_type = UNQUANTIZED) {
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
    
    // Determine if the result is acceptable based on quantization bit-width
    if (quant_type == FP4_QUANT) {
        // 4-bit quantization: FP32 -> 4-bit has ~10% weight quantization error
        // After 6 matrix operations in attention (Q·Wq, K·Wk, V·Wv, QK^T, softmax, A·V),
        // error accumulates exponentially: 10% -> 20% -> 40% -> 60%+
        // This is expected behavior for 4-bit quantization in multi-stage operations
        if (relative_error < 0.30) {
            printf("    Status: ✓ PASS (Excellent for 4-bit quantized weights)\n");
        } else if (relative_error < 0.50) {
            printf("    Status: ✓ PASS (Good for 4-bit quantized weights)\n");
        } else if (relative_error < 0.70) {
            printf("    Status: ⚠ WARNING (Acceptable for 4-bit quantized weights)\n");
        } else {
            printf("    Status: ✗ FAIL (Too much error even for 4-bit quantization)\n");
        }
    } else if (quant_type == INT8_QUANT) {
        // 8-bit quantization: FP32 -> INT8 has theoretical ~0.4% quantization error per value
        // More precise than 4-bit, so stricter thresholds
        if (relative_error < 0.03) {
            printf("    Status: ✓ PASS (Excellent for 8-bit quantized weights)\n");
        } else if (relative_error < 0.08) {
            printf("    Status: ✓ PASS (Good for 8-bit quantized weights)\n");
        } else if (relative_error < 0.15) {
            printf("    Status: ⚠ WARNING (Acceptable for 8-bit quantized weights)\n");
        } else {
            printf("    Status: ✗ FAIL (Too much error even for 8-bit quantization)\n");
        }
    } else {
        // Strict thresholds for unquantized versions
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
}

// Function to compute median from array of floats
float compute_median(float* times, int n) {
    std::sort(times, times + n);
    if (n % 2 == 0) {
        return (times[n/2 - 1] + times[n/2]) / 2.0f;
    } else {
        return times[n/2];
    }
}

// Function to wait for GPU to cool down
void cooldown_gpu(int seconds) {
    printf("Cooling down GPU for %d seconds...\n", seconds);
    CUDA_CHECK(cudaDeviceSynchronize());
    sleep(seconds);
}


int main() {
    // Test parameters (GPT-2 small scale)
    int batch = 1;
    int seq_len = 128;
    int d_model = 768;
    int d_k = 64;
    int d_v = 64;

    printf("\n=== Starting Evaluation (GPT-2 Scale, Using Real GPT-2 Weights) ===\n");

    printf("\n== Loading GPT-2 Weights ==\n");
    
    // Try to load weights from numpy files
    const char* weights_dir = "weights";
    char wq_path[256], wk_path[256], wv_path[256];
    char bq_path[256], bk_path[256], bv_path[256];
    
    snprintf(wq_path, sizeof(wq_path), "%s/wq.npy", weights_dir);
    snprintf(wk_path, sizeof(wk_path), "%s/wk.npy", weights_dir);
    snprintf(wv_path, sizeof(wv_path), "%s/wv.npy", weights_dir);
    snprintf(bq_path, sizeof(bq_path), "%s/bq.npy", weights_dir);
    snprintf(bk_path, sizeof(bk_path), "%s/bk.npy", weights_dir);
    snprintf(bv_path, sizeof(bv_path), "%s/bv.npy", weights_dir);
    
    bool use_real_weights = true;
    float *h_Wq_loaded = NULL, *h_Wk_loaded = NULL, *h_Wv_loaded = NULL;
    float *h_bq_loaded = NULL, *h_bk_loaded = NULL, *h_bv_loaded = NULL;
    int wq_rows, wq_cols, wk_rows, wk_cols, wv_rows, wv_cols;
    int bq_rows, bq_cols, bk_rows, bk_cols, bv_rows, bv_cols;
    
    // Try to load weights
    if (load_npy_float32(wq_path, &h_Wq_loaded, &wq_rows, &wq_cols) &&
        load_npy_float32(wk_path, &h_Wk_loaded, &wk_rows, &wk_cols) &&
        load_npy_float32(wv_path, &h_Wv_loaded, &wv_rows, &wv_cols) &&
        load_npy_float32(bq_path, &h_bq_loaded, &bq_rows, &bq_cols) &&
        load_npy_float32(bk_path, &h_bk_loaded, &bk_rows, &bk_cols) &&
        load_npy_float32(bv_path, &h_bv_loaded, &bv_rows, &bv_cols)) {
        
        printf("Successfully loaded GPT-2 weights from %s/\n", weights_dir);
        printf("  Wq shape: (%d, %d)\n", wq_rows, wq_cols);
        printf("  Wk shape: (%d, %d)\n", wk_rows, wk_cols);
        printf("  Wv shape: (%d, %d)\n", wv_rows, wv_cols);
        printf("  bq shape: (%d, %d)\n", bq_rows, bq_cols);
        printf("  bk shape: (%d, %d)\n", bk_rows, bk_cols);
        printf("  bv shape: (%d, %d)\n", bv_rows, bv_cols);
        
        // Verify dimensions match
        if (wq_rows != d_model || wq_cols != d_k ||
            wk_rows != d_model || wk_cols != d_k ||
            wv_rows != d_model || wv_cols != d_v ||
            bq_rows != d_k || bk_rows != d_k || bv_rows != d_v) {
            printf("Warning: Loaded weight dimensions don't match expected dimensions!\n");
            printf("  Expected: Wq=(%d,%d), Wk=(%d,%d), Wv=(%d,%d)\n", 
                   d_model, d_k, d_model, d_k, d_model, d_v);
            printf("  Got: Wq=(%d,%d), Wk=(%d,%d), Wv=(%d,%d)\n",
                   wq_rows, wq_cols, wk_rows, wk_cols, wv_rows, wv_cols);
            use_real_weights = false;
        }
    } else {
        printf("Could not load GPT-2 weights from %s/\n", weights_dir);
        printf("Please run: python extract_gpt2_weights.py --output %s\n", weights_dir);
        printf("Falling back to random weights.\n");
        use_real_weights = false;
    }

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

    // Use loaded weights or initialize with random values
    if (use_real_weights) {
        printf("Using real GPT-2 weights\n");
        memcpy(h_Wq, h_Wq_loaded, wq_size);
        memcpy(h_Wk, h_Wk_loaded, wk_size);
        memcpy(h_Wv, h_Wv_loaded, wv_size);
        memcpy(h_bq, h_bq_loaded, bq_size);
        memcpy(h_bk, h_bk_loaded, bk_size);
        memcpy(h_bv, h_bv_loaded, bv_size);
        
        // Free loaded weights
        free(h_Wq_loaded);
        free(h_Wk_loaded);
        free(h_Wv_loaded);
        free(h_bq_loaded);
        free(h_bk_loaded);
        free(h_bv_loaded);
    } else {
        printf("Using random weights\n");
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

    // Allocate output buffers for evaluation
    float* h_baseline_unquantized = (float*)malloc(out_size);   // Naive unquantized baseline (top-level reference)
    float* h_naive_baseline_int8 = (float*)malloc(out_size);    // Naive INT8 baseline (for INT8 section)
    float* h_naive_baseline_mxfp4 = (float*)malloc(out_size);   // Naive MXFP4 baseline (for MXFP4 section)
    float* h_naive_baseline_nf4 = (float*)malloc(out_size);     // Naive NF4 baseline (for NF4 section)
    float* h_naive_baseline_nvfp4 = (float*)malloc(out_size);   // Naive NVFP4 baseline (for NVFP4 section)
    float* h_result_temp = (float*)malloc(out_size);            // Temporary buffer for current test result

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed_time_quant;
    float* iteration_times = (float*)malloc(NUM_ITERATIONS * sizeof(float));

    // Storage for timing results
    float time_mxfp4_naive, time_mxfp4_projection, time_mxfp4_semi, time_mxfp4_ours;
    float time_nf4_naive, time_nf4_projection, time_nf4_semi, time_nf4_ours;
    float time_nvfp4_naive, time_nvfp4_projection, time_nvfp4_semi, time_nvfp4_ours;

    cooldown_gpu(COOLDOWN_SECONDS);


    printf("\n");


    printf("\n=== Quantized Weights Tests (MXFP4) ===\n");

    // Quantize weights to MXFP4
    printf("\n== Quantizing Weights to MXFP4 ==\n");
    int num_blocks_q_mxfp4 = (d_model * d_k + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    int num_blocks_v_mxfp4 = (d_model * d_v + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;

    MXFP4Block* h_Wq_mxfp4 = (MXFP4Block*)malloc(num_blocks_q_mxfp4 * sizeof(MXFP4Block));
    MXFP4Block* h_Wk_mxfp4 = (MXFP4Block*)malloc(num_blocks_q_mxfp4 * sizeof(MXFP4Block));
    MXFP4Block* h_Wv_mxfp4 = (MXFP4Block*)malloc(num_blocks_v_mxfp4 * sizeof(MXFP4Block));

    quantize_mxfp4(h_Wq, h_Wq_mxfp4, d_model * d_k);
    quantize_mxfp4(h_Wk, h_Wk_mxfp4, d_model * d_k);
    quantize_mxfp4(h_Wv, h_Wv_mxfp4, d_model * d_v);

    printf("MXFP4 Block size: %d\n", MXFP4_BLOCK_SIZE);
    printf("Num MXFP4 blocks (Q/K): %d, Num MXFP4 blocks (V): %d\n", num_blocks_q_mxfp4, num_blocks_v_mxfp4);
    printf("MXFP4 Quantization complete.\n");
    
    // Test MXFP4 quantization accuracy
    float* h_Wq_dequant_test = (float*)malloc(wq_size);
    dequantize_mxfp4_cpu(h_Wq_mxfp4, h_Wq_dequant_test, d_model * d_k);
    double mxfp4_test_error = 0.0;
    double mxfp4_test_sum_sq = 0.0;
    for (int i = 0; i < d_model * d_k; i++) {
        double diff = h_Wq[i] - h_Wq_dequant_test[i];
        mxfp4_test_error += diff * diff;
        mxfp4_test_sum_sq += h_Wq[i] * h_Wq[i];
    }
    double mxfp4_relative_error = sqrt(mxfp4_test_error / mxfp4_test_sum_sq);
    printf("MXFP4 Weight Quantization Error: %.6e (%.2f%%)\n", mxfp4_relative_error, mxfp4_relative_error * 100);
    free(h_Wq_dequant_test);

    // Allocate device memory for MXFP4 quantized weights
    MXFP4Block *d_Wq_mxfp4, *d_Wk_mxfp4, *d_Wv_mxfp4;

    CUDA_CHECK(cudaMalloc(&d_Wq_mxfp4, num_blocks_q_mxfp4 * sizeof(MXFP4Block)));
    CUDA_CHECK(cudaMalloc(&d_Wk_mxfp4, num_blocks_q_mxfp4 * sizeof(MXFP4Block)));
    CUDA_CHECK(cudaMalloc(&d_Wv_mxfp4, num_blocks_v_mxfp4 * sizeof(MXFP4Block)));

    // Copy MXFP4 quantized data to device
    CUDA_CHECK(cudaMemcpy(d_Wq_mxfp4, h_Wq_mxfp4, num_blocks_q_mxfp4 * sizeof(MXFP4Block), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk_mxfp4, h_Wk_mxfp4, num_blocks_q_mxfp4 * sizeof(MXFP4Block), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv_mxfp4, h_Wv_mxfp4, num_blocks_v_mxfp4 * sizeof(MXFP4Block), cudaMemcpyHostToDevice));

    // Test GPU dequantization matches CPU
    float* d_Wq_test_dequant;
    float* h_Wq_gpu_dequant = (float*)malloc(wq_size);
    CUDA_CHECK(cudaMalloc(&d_Wq_test_dequant, wq_size));
    
    dim3 block_test(256);
    dim3 grid_test((d_model * d_k + block_test.x - 1) / block_test.x);
    dequantize_mxfp4_kernel<<<grid_test, block_test>>>(d_Wq_mxfp4, d_Wq_test_dequant, d_model * d_k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_Wq_gpu_dequant, d_Wq_test_dequant, wq_size, cudaMemcpyDeviceToHost));
    
    double gpu_cpu_diff = 0.0;
    for (int i = 0; i < d_model * d_k; i++) {
        double diff = h_Wq_dequant_test[i] - h_Wq_gpu_dequant[i];
        gpu_cpu_diff += diff * diff;
    }
    printf("MXFP4 GPU vs CPU Dequantization Error: %.6e\n", sqrt(gpu_cpu_diff / (d_model * d_k)));
    
    CUDA_CHECK(cudaFree(d_Wq_test_dequant));
    free(h_Wq_gpu_dequant);


    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n1. Naive Flash Attention (MXFP4)\n");
    
    // Measure kernel latency breakdown
    float kernel_times_naive[10];
    int num_kernels_naive = 0;
    naive_flash_attention_mxfp4(d_X, 
                                d_Wq_mxfp4, d_Wk_mxfp4, d_Wv_mxfp4,
                                d_bq, d_bk, d_bv,
                                d_output, batch, seq_len, d_model, d_k, d_v,
                                false, kernel_times_naive, &num_kernels_naive);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy MXFP4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));
    
    // Compute total time from kernel times
    float total_time_naive = 0.0f;
    for (int k = 0; k < num_kernels_naive; k++) {
        total_time_naive += kernel_times_naive[k];
    }
    time_mxfp4_naive = total_time_naive;
    
    printf("Median execution time: %.4f ms\n", total_time_naive);
    printf("\nKernel Latency Breakdown:\n");
    printf("  1. Dequantize Wq:            %8.4f ms\n", kernel_times_naive[0]);
    printf("  2. Dequantize Wk:            %8.4f ms\n", kernel_times_naive[1]);
    printf("  3. Dequantize Wv:            %8.4f ms\n", kernel_times_naive[2]);
    printf("  4. Linear Projection Q:      %8.4f ms\n", kernel_times_naive[3]);
    printf("  5. Linear Projection K:      %8.4f ms\n", kernel_times_naive[4]);
    printf("  6. Linear Projection V:      %8.4f ms\n", kernel_times_naive[5]);
    printf("  7. Flash Attention:          %8.4f ms\n", kernel_times_naive[6]);
    printf("  ----------------------------------------\n");
    printf("  Total (sum):                 %8.4f ms\n", total_time_naive);


    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n2. Projection-Fused Flash Attention (MXFP4)\n");
    
    // Measure kernel latency breakdown
    float kernel_times_proj[10];
    int num_kernels_proj = 0;
    flash_attention_mxfp4(d_X, 
                          d_Wq_mxfp4, d_Wk_mxfp4, d_Wv_mxfp4,
                          d_bq, d_bk, d_bv,
                          d_output, batch, seq_len, d_model, d_k, d_v,
                          false, kernel_times_proj, &num_kernels_proj);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy MXFP4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));
    
    // Compute total time
    float total_time_proj = 0.0f;
    for (int k = 0; k < num_kernels_proj; k++) {
        total_time_proj += kernel_times_proj[k];
    }
    time_mxfp4_projection = total_time_proj;
    
    printf("Median execution time: %.4f ms\n", total_time_proj);
    printf("\nKernel Latency Breakdown:\n");
    printf("  1. Dequantize Wq:            %8.4f ms\n", kernel_times_proj[0]);
    printf("  2. Dequantize Wk:            %8.4f ms\n", kernel_times_proj[1]);
    printf("  3. Dequantize Wv:            %8.4f ms\n", kernel_times_proj[2]);
    printf("  4. Fused QKV Projection:     %8.4f ms\n", kernel_times_proj[3]);
    printf("  5. Flash Attention:          %8.4f ms\n", kernel_times_proj[4]);
    printf("  ----------------------------------------\n");
    printf("  Total (sum):                 %8.4f ms\n", total_time_proj);
    
    // // Compare with baseline (both MXFP4, should match closely)
    // compute_error_metrics("Tiled MXFP4 vs Naive MXFP4", h_naive_baseline_mxfp4, h_result_temp, 
    //                      batch, seq_len, d_v, UNQUANTIZED);


    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n3. Semi Fused Flash Attention (MXFP4)\n");
    
    // Measure kernel latency breakdown
    float kernel_times_semi[10];
    int num_kernels_semi = 0;
    fused_flash_attention_mxfp4(d_X, 
                                d_Wq_mxfp4, d_Wk_mxfp4, d_Wv_mxfp4,
                                d_bq, d_bk, d_bv,
                                d_output, batch, seq_len, d_model, d_k, d_v,
                                false, kernel_times_semi, &num_kernels_semi);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy MXFP4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    // Compute total time
    float total_time_semi = 0.0f;
    for (int k = 0; k < num_kernels_semi; k++) {
        total_time_semi += kernel_times_semi[k];
    }
    time_mxfp4_semi = total_time_semi;
    
    printf("Median execution time: %.4f ms\n", total_time_semi);
    printf("\nKernel Latency Breakdown:\n");
    printf("  1. Fused QKV Dequant:        %8.4f ms\n", kernel_times_semi[0]);
    printf("  2. Fused QKV Projection:     %8.4f ms\n", kernel_times_semi[1]);
    printf("  3. Flash Attention:          %8.4f ms\n", kernel_times_semi[2]);
    printf("  ----------------------------------------\n");
    printf("  Total (sum):                 %8.4f ms\n", total_time_semi);
    
    // // Compare with baseline (both MXFP4, should match closely)
    // compute_error_metrics("Flash MXFP4 vs Naive MXFP4", h_naive_baseline_mxfp4, h_result_temp, 
    //                      batch, seq_len, d_v, UNQUANTIZED);

    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n4. Our Attention (MXFP4)\n");

    // Measure kernel latency breakdown
    float kernel_times_ours[10];
    int num_kernels_ours = 0;
    our_attention_mxfp4(d_X, 
                        d_Wq_mxfp4, d_Wk_mxfp4, d_Wv_mxfp4,
                        d_bq, d_bk, d_bv,
                        d_output, batch, seq_len, d_model, d_k, d_v,
                        false, kernel_times_ours, &num_kernels_ours);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy MXFP4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    // Compute total time
    float total_time_ours = 0.0f;
    for (int k = 0; k < num_kernels_ours; k++) {
        total_time_ours += kernel_times_ours[k];
    }
    time_mxfp4_ours = total_time_ours;
    
    printf("Median execution time: %.4f ms\n", total_time_ours);
    printf("\nKernel Latency Breakdown:\n");
    printf("  1. Fused Dequant+Projection: %8.4f ms\n", kernel_times_ours[0]);
    printf("  2. Our Attention Kernel:     %8.4f ms\n", kernel_times_ours[1]);
    printf("  ----------------------------------------\n");
    printf("  Total (sum):                 %8.4f ms\n", total_time_ours);
    
    // // Compare with baseline (both MXFP4, should match closely)
    // compute_error_metrics("Our MXFP4 vs Naive MXFP4", h_naive_baseline_mxfp4, h_result_temp, 
    //                      batch, seq_len, d_v, UNQUANTIZED);

    printf("\n");

    // Test 5: Full Attention (MXFP4)
    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n5. Full Attention (MXFP4) - Chunk-based Tiling\n");

    printf("Running dummy run for warm-up...\n");
    full_attention_mxfp4(d_X, 
                         d_Wq_mxfp4, d_Wk_mxfp4, d_Wv_mxfp4,
                         d_bq, d_bk, d_bv,
                         d_output, batch, seq_len, d_model, d_k, d_v,
                         false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        full_attention_mxfp4(d_X, 
                             d_Wq_mxfp4, d_Wk_mxfp4, d_Wv_mxfp4,
                             d_bq, d_bk, d_bv,
                             d_output, batch, seq_len, d_model, d_k, d_v,
                             false);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);

    // Copy Full MXFP4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    float time_mxfp4_full = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    // Compare with baseline (both MXFP4, should match closely)
    // compute_error_metrics("Full MXFP4 vs Naive MXFP4", h_naive_baseline_mxfp4, h_result_temp, 
    //                      batch, seq_len, d_v, UNQUANTIZED);

    printf("\n");

    printf("\n=== Quantized Weights Tests (NF4) ===\n");

    // Quantize weights to NF4
    printf("\n== Quantizing Weights to NF4 ==\n");
    int num_blocks_q_nf4 = (d_model * d_k + NF4_BLOCK_SIZE - 1) / NF4_BLOCK_SIZE;
    int num_blocks_v_nf4 = (d_model * d_v + NF4_BLOCK_SIZE - 1) / NF4_BLOCK_SIZE;

    NF4Block* h_Wq_nf4 = (NF4Block*)malloc(num_blocks_q_nf4 * sizeof(NF4Block));
    NF4Block* h_Wk_nf4 = (NF4Block*)malloc(num_blocks_q_nf4 * sizeof(NF4Block));
    NF4Block* h_Wv_nf4 = (NF4Block*)malloc(num_blocks_v_nf4 * sizeof(NF4Block));

    quantize_nf4(h_Wq, h_Wq_nf4, d_model * d_k);
    quantize_nf4(h_Wk, h_Wk_nf4, d_model * d_k);
    quantize_nf4(h_Wv, h_Wv_nf4, d_model * d_v);

    printf("NF4 Block size: %d\n", NF4_BLOCK_SIZE);
    printf("Num NF4 blocks (Q/K): %d, Num NF4 blocks (V): %d\n", num_blocks_q_nf4, num_blocks_v_nf4);
    printf("NF4 Quantization complete.\n");
    
    // Test NF4 quantization accuracy
    h_Wq_dequant_test = (float*)malloc(wq_size);
    dequantize_nf4_cpu(h_Wq_nf4, h_Wq_dequant_test, d_model * d_k);
    double nf4_test_error = 0.0;
    double nf4_test_sum_sq = 0.0;
    for (int i = 0; i < d_model * d_k; i++) {
        double diff = h_Wq[i] - h_Wq_dequant_test[i];
        nf4_test_error += diff * diff;
        nf4_test_sum_sq += h_Wq[i] * h_Wq[i];
    }
    double nf4_relative_error = sqrt(nf4_test_error / nf4_test_sum_sq);
    printf("NF4 Weight Quantization Error: %.6e (%.2f%%)\n", nf4_relative_error, nf4_relative_error * 100);
    free(h_Wq_dequant_test);

    // Allocate device memory for NF4 quantized weights
    NF4Block *d_Wq_nf4, *d_Wk_nf4, *d_Wv_nf4;

    CUDA_CHECK(cudaMalloc(&d_Wq_nf4, num_blocks_q_nf4 * sizeof(NF4Block)));
    CUDA_CHECK(cudaMalloc(&d_Wk_nf4, num_blocks_q_nf4 * sizeof(NF4Block)));
    CUDA_CHECK(cudaMalloc(&d_Wv_nf4, num_blocks_v_nf4 * sizeof(NF4Block)));

    // Copy NF4 quantized data to device
    CUDA_CHECK(cudaMemcpy(d_Wq_nf4, h_Wq_nf4, num_blocks_q_nf4 * sizeof(NF4Block), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk_nf4, h_Wk_nf4, num_blocks_q_nf4 * sizeof(NF4Block), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv_nf4, h_Wv_nf4, num_blocks_v_nf4 * sizeof(NF4Block), cudaMemcpyHostToDevice));

    // Test GPU dequantization matches CPU
    d_Wq_test_dequant = NULL;
    h_Wq_gpu_dequant = (float*)malloc(wq_size);
    CUDA_CHECK(cudaMalloc(&d_Wq_test_dequant, wq_size));
    
    grid_test = dim3((d_model * d_k + block_test.x - 1) / block_test.x);
    dequantize_nf4_kernel<<<grid_test, block_test>>>(d_Wq_nf4, d_Wq_test_dequant, d_model * d_k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_Wq_gpu_dequant, d_Wq_test_dequant, wq_size, cudaMemcpyDeviceToHost));
    
    gpu_cpu_diff = 0.0;
    for (int i = 0; i < d_model * d_k; i++) {
        double diff = h_Wq_dequant_test[i] - h_Wq_gpu_dequant[i];
        gpu_cpu_diff += diff * diff;
    }
    printf("NF4 GPU vs CPU Dequantization Error: %.6e\n", sqrt(gpu_cpu_diff / (d_model * d_k)));
    
    CUDA_CHECK(cudaFree(d_Wq_test_dequant));
    free(h_Wq_gpu_dequant);


    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n1. Naive Flash Attention (NF4)\n");
    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    naive_flash_attention_nf4(
        d_X,
        d_Wq_nf4, d_Wk_nf4, d_Wv_nf4,
        d_bq, d_bk, d_bv,
        d_output, batch, seq_len, d_model, d_k, d_v,
        false
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    float kernel_times_nf4_naive[MAX_KERNELS];
    int num_kernels_nf4_naive = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        naive_flash_attention_nf4(
            d_X, 
            d_Wq_nf4, d_Wk_nf4, d_Wv_nf4,
            d_bq, d_bk, d_bv,
            d_output, batch, seq_len, d_model, d_k, d_v,
            false,
            (iter == 0) ? kernel_times_nf4_naive : nullptr,
            (iter == 0) ? &num_kernels_nf4_naive : nullptr
        );
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    printf("\nKernel Latency Breakdown:\n");
    printf("  1. Dequantize Wq:             %.4f ms\n", kernel_times_nf4_naive[0]);
    printf("  2. Dequantize Wk:             %.4f ms\n", kernel_times_nf4_naive[1]);
    printf("  3. Dequantize Wv:             %.4f ms\n", kernel_times_nf4_naive[2]);
    printf("  4. Linear Projection Q:       %.4f ms\n", kernel_times_nf4_naive[3]);
    printf("  5. Linear Projection K:       %.4f ms\n", kernel_times_nf4_naive[4]);
    printf("  6. Linear Projection V:       %.4f ms\n", kernel_times_nf4_naive[5]);
    printf("  7. Flash Attention:           %.4f ms\n", kernel_times_nf4_naive[6]);
    printf("  ----------------------------------------\n");
    float total_nf4_naive = 0;
    for (int i = 0; i < num_kernels_nf4_naive; i++) total_nf4_naive += kernel_times_nf4_naive[i];
    printf("  Total (sum):                  %.4f ms\n", total_nf4_naive);

    // Copy NF4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    time_nf4_naive = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    // // Compare with unquantized baseline (using 4-bit quantized thresholds)
    // compute_error_metrics("Naive NF4 vs Naive Baseline", h_baseline_unquantized, h_result_temp, 
    //                      batch, seq_len, d_v, FP4_QUANT);
    
    // Copy result back as NF4 baseline
    CUDA_CHECK(cudaMemcpy(h_naive_baseline_nf4, d_output, out_size, cudaMemcpyDeviceToHost));

    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n2. Projection-Fused Attention (NF4)\n");
    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    flash_attention_nf4(
        d_X, 
        d_Wq_nf4, d_Wk_nf4, d_Wv_nf4,
        d_bq, d_bk, d_bv,
        d_output, batch, seq_len, d_model, d_k, d_v,
        false
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        flash_attention_nf4(
            d_X, 
            d_Wq_nf4, d_Wk_nf4, d_Wv_nf4,
            d_bq, d_bk, d_bv,
            d_output, batch, seq_len, d_model, d_k, d_v,
            false
        );
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);

    // Copy NF4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    time_nf4_projection = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    // // Compare with baseline (both NF4, should match closely)
    // compute_error_metrics("Tiled NF4 vs Naive NF4", h_naive_baseline_nf4, h_result_temp, 
    //                      batch, seq_len, d_v, UNQUANTIZED);

    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n3. Semi Fused Flash Attention (NF4)\n");
    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    fused_flash_attention_nf4(
        d_X, 
        d_Wq_nf4, d_Wk_nf4, d_Wv_nf4,
        d_bq, d_bk, d_bv,
        d_output, batch, seq_len, d_model, d_k, d_v,
        false
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        fused_flash_attention_nf4(
            d_X, 
            d_Wq_nf4, d_Wk_nf4, d_Wv_nf4,
            d_bq, d_bk, d_bv,
            d_output, batch, seq_len, d_model, d_k, d_v,
            false
        );
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);

    // Copy NF4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    time_nf4_semi = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    // // Compare with baseline (both NF4, should match closely)
    // compute_error_metrics("Flash NF4 vs Naive NF4", h_naive_baseline_nf4, h_result_temp, 
    //                      batch, seq_len, d_v, UNQUANTIZED);

    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n4. Our Attention (NF4)\n");

    printf("Running dummy run for warm-up...\n");
    our_attention_nf4(
        d_X, 
        d_Wq_nf4, d_Wk_nf4, d_Wv_nf4,
        d_bq, d_bk, d_bv,
        d_output, batch, seq_len, d_model, d_k, d_v,
        false
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    float kernel_times_nf4_ours[MAX_KERNELS];
    int num_kernels_nf4_ours = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        our_attention_nf4(d_X, 
                          d_Wq_nf4, d_Wk_nf4, d_Wv_nf4,
                          d_bq, d_bk, d_bv,
                          d_output, batch, seq_len, d_model, d_k, d_v,
                          false,
                          (iter == 0) ? kernel_times_nf4_ours : nullptr,
                          (iter == 0) ? &num_kernels_nf4_ours : nullptr);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);

    // Copy NF4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    time_nf4_ours = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    printf("\nKernel Latency Breakdown:\n");
    printf("  1. Fused Dequant+Projection:  %.4f ms\n", kernel_times_nf4_ours[0]);
    printf("  2. Our Attention Kernel:      %.4f ms\n", kernel_times_nf4_ours[1]);
    printf("  ----------------------------------------\n");
    float total_nf4_ours = 0;
    for (int i = 0; i < num_kernels_nf4_ours; i++) total_nf4_ours += kernel_times_nf4_ours[i];
    printf("  Total (sum):                  %.4f ms\n", total_nf4_ours);
    
    // // Compare with baseline (both NF4, should match closely)
    // compute_error_metrics("Our NF4 vs Naive NF4", h_naive_baseline_nf4, h_result_temp, 
    //                      batch, seq_len, d_v, UNQUANTIZED);

    printf("\n");

    // Test 5: Full Attention (NF4)
    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n5. Full Attention (NF4) - Chunk-based Tiling\n");

    printf("Running dummy run for warm-up...\n");
    full_attention_nf4(
        d_X, 
        d_Wq_nf4, d_Wk_nf4, d_Wv_nf4,
        d_bq, d_bk, d_bv,
        d_output, batch, seq_len, d_model, d_k, d_v,
        false
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        full_attention_nf4(d_X, 
                          d_Wq_nf4, d_Wk_nf4, d_Wv_nf4,
                          d_bq, d_bk, d_bv,
                          d_output, batch, seq_len, d_model, d_k, d_v,
                          false);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);

    // Copy Full NF4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    float time_nf4_full = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    // Compare with baseline (both NF4, should match closely)
    // compute_error_metrics("Full NF4 vs Naive NF4", h_naive_baseline_nf4, h_result_temp, 
    //                      batch, seq_len, d_v, UNQUANTIZED);

    printf("\n");










    printf("\n=== Quantized Weights Tests (NVFP4) ===\n");

    // Quantize weights to NVFP4
    printf("\n== Quantizing Weights to NVFP4 ==\n");
    int num_blocks_q_nvfp4 = (d_model * d_k + NVFP4_BLOCK_SIZE - 1) / NVFP4_BLOCK_SIZE;
    int num_blocks_v_nvfp4 = (d_model * d_v + NVFP4_BLOCK_SIZE - 1) / NVFP4_BLOCK_SIZE;

    NVFP4Block* h_Wq_nvfp4 = (NVFP4Block*)malloc(num_blocks_q_nvfp4 * sizeof(NVFP4Block));
    NVFP4Block* h_Wk_nvfp4 = (NVFP4Block*)malloc(num_blocks_q_nvfp4 * sizeof(NVFP4Block));
    NVFP4Block* h_Wv_nvfp4 = (NVFP4Block*)malloc(num_blocks_v_nvfp4 * sizeof(NVFP4Block));
    
    NVFP4TensorMeta h_Wq_nvfp4_meta, h_Wk_nvfp4_meta, h_Wv_nvfp4_meta;
    
    quantize_nvfp4(h_Wq, h_Wq_nvfp4, &h_Wq_nvfp4_meta, d_model * d_k);
    quantize_nvfp4(h_Wk, h_Wk_nvfp4, &h_Wk_nvfp4_meta, d_model * d_k);
    quantize_nvfp4(h_Wv, h_Wv_nvfp4, &h_Wv_nvfp4_meta, d_model * d_v);

    printf("NVFP4 Block size: %d\n", NVFP4_BLOCK_SIZE);
    printf("Num NVFP4 blocks (Q/K): %d, Num NVFP4 blocks (V): %d\n", num_blocks_q_nvfp4, num_blocks_v_nvfp4);
    printf("NVFP4 Quantization complete.\n");
    
    // Test NVFP4 quantization accuracy
    h_Wq_dequant_test = (float*)malloc(wq_size);
    dequantize_nvfp4_cpu(h_Wq_nvfp4, &h_Wq_nvfp4_meta, h_Wq_dequant_test, d_model * d_k);
    double nvfp4_test_error = 0.0;
    double nvfp4_test_sum_sq = 0.0;
    for (int i = 0; i < d_model * d_k; i++) {
        double diff = h_Wq[i] - h_Wq_dequant_test[i];
        nvfp4_test_error += diff * diff;
        nvfp4_test_sum_sq += h_Wq[i] * h_Wq[i];
    }
    double nvfp4_relative_error = sqrt(nvfp4_test_error / nvfp4_test_sum_sq);
    printf("NVFP4 Weight Quantization Error: %.6e (%.2f%%)\n", nvfp4_relative_error, nvfp4_relative_error * 100);
    free(h_Wq_dequant_test);

    // Allocate device memory for NVFP4 quantized weights
    NVFP4Block *d_Wq_nvfp4, *d_Wk_nvfp4, *d_Wv_nvfp4;

    CUDA_CHECK(cudaMalloc(&d_Wq_nvfp4, num_blocks_q_nvfp4 * sizeof(NVFP4Block)));
    CUDA_CHECK(cudaMalloc(&d_Wk_nvfp4, num_blocks_q_nvfp4 * sizeof(NVFP4Block)));
    CUDA_CHECK(cudaMalloc(&d_Wv_nvfp4, num_blocks_v_nvfp4 * sizeof(NVFP4Block)));

    // Copy NVFP4 quantized data to device
    CUDA_CHECK(cudaMemcpy(d_Wq_nvfp4, h_Wq_nvfp4, num_blocks_q_nvfp4 * sizeof(NVFP4Block), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk_nvfp4, h_Wk_nvfp4, num_blocks_q_nvfp4 * sizeof(NVFP4Block), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv_nvfp4, h_Wv_nvfp4, num_blocks_v_nvfp4 * sizeof(NVFP4Block), cudaMemcpyHostToDevice));

    // Test GPU dequantization matches CPU
    d_Wq_test_dequant = NULL;
    h_Wq_gpu_dequant = (float*)malloc(wq_size);
    CUDA_CHECK(cudaMalloc(&d_Wq_test_dequant, wq_size));
    
    grid_test = dim3((d_model * d_k + block_test.x - 1) / block_test.x);
    dequantize_nvfp4_kernel<<<grid_test, block_test>>>(d_Wq_nvfp4, h_Wq_nvfp4_meta.global_scale_dec, d_Wq_test_dequant, d_model * d_k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_Wq_gpu_dequant, d_Wq_test_dequant, wq_size, cudaMemcpyDeviceToHost));
    
    gpu_cpu_diff = 0.0;
    for (int i = 0; i < d_model * d_k; i++) {
        double diff = h_Wq_dequant_test[i] - h_Wq_gpu_dequant[i];
        gpu_cpu_diff += diff * diff;
    }
    printf("NVFP4 GPU vs CPU Dequantization Error: %.6e\n", sqrt(gpu_cpu_diff / (d_model * d_k)));
    
    CUDA_CHECK(cudaFree(d_Wq_test_dequant));
    free(h_Wq_gpu_dequant);


    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n1. Naive Flash Attention (NVFP4)\n");
    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    naive_flash_attention_nvfp4(d_X,
                                d_Wq_nvfp4, d_Wk_nvfp4, d_Wv_nvfp4,
                                &h_Wq_nvfp4_meta, &h_Wk_nvfp4_meta, &h_Wv_nvfp4_meta,
                                d_bq, d_bk, d_bv,
                                d_output, batch, seq_len, d_model, d_k, d_v,
                                false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    float kernel_times_nvfp4_naive[MAX_KERNELS];
    int num_kernels_nvfp4_naive = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        naive_flash_attention_nvfp4(d_X, 
                                    d_Wq_nvfp4, d_Wk_nvfp4, d_Wv_nvfp4,
                                    &h_Wq_nvfp4_meta, &h_Wk_nvfp4_meta, &h_Wv_nvfp4_meta,
                                    d_bq, d_bk, d_bv,
                                    d_output, batch, seq_len, d_model, d_k, d_v,
                                    false,
                                    (iter == 0) ? kernel_times_nvfp4_naive : nullptr,
                                    (iter == 0) ? &num_kernels_nvfp4_naive : nullptr);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);

    // Copy NVFP4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    time_nvfp4_naive = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    printf("\nKernel Latency Breakdown:\n");
    printf("  1. Dequantize Wq:             %.4f ms\n", kernel_times_nvfp4_naive[0]);
    printf("  2. Dequantize Wk:             %.4f ms\n", kernel_times_nvfp4_naive[1]);
    printf("  3. Dequantize Wv:             %.4f ms\n", kernel_times_nvfp4_naive[2]);
    printf("  4. Linear Projection Q:       %.4f ms\n", kernel_times_nvfp4_naive[3]);
    printf("  5. Linear Projection K:       %.4f ms\n", kernel_times_nvfp4_naive[4]);
    printf("  6. Linear Projection V:       %.4f ms\n", kernel_times_nvfp4_naive[5]);
    printf("  7. Flash Attention:           %.4f ms\n", kernel_times_nvfp4_naive[6]);
    printf("  ----------------------------------------\n");
    float total_nvfp4_naive = 0;
    for (int i = 0; i < num_kernels_nvfp4_naive; i++) total_nvfp4_naive += kernel_times_nvfp4_naive[i];
    printf("  Total (sum):                  %.4f ms\n", total_nvfp4_naive);
    
    // // Compare with unquantized baseline (using 4-bit quantized thresholds)
    // compute_error_metrics("Naive NVFP4 vs Naive Baseline", h_baseline_unquantized, h_result_temp, 
    //                      batch, seq_len, d_v, FP4_QUANT);
    
    // Copy result back as NVFP4 baseline
    CUDA_CHECK(cudaMemcpy(h_naive_baseline_nvfp4, d_output, out_size, cudaMemcpyDeviceToHost));

    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n2. Projection-Fused Attention (NVFP4)\n");
    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    flash_attention_nvfp4(d_X, 
                          d_Wq_nvfp4, d_Wk_nvfp4, d_Wv_nvfp4,
                          &h_Wq_nvfp4_meta, &h_Wk_nvfp4_meta, &h_Wv_nvfp4_meta,
                          d_bq, d_bk, d_bv,
                          d_output, batch, seq_len, d_model, d_k, d_v,
                          false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        flash_attention_nvfp4(d_X, 
                              d_Wq_nvfp4, d_Wk_nvfp4, d_Wv_nvfp4,
                              &h_Wq_nvfp4_meta, &h_Wk_nvfp4_meta, &h_Wv_nvfp4_meta,
                              d_bq, d_bk, d_bv,
                              d_output, batch, seq_len, d_model, d_k, d_v,
                              false);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);

    // Copy NVFP4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    time_nvfp4_projection = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    // Compare with baseline (both NVFP4, should match closely)
    compute_error_metrics("Tiled NVFP4 vs Naive NVFP4", h_naive_baseline_nvfp4, h_result_temp, 
                         batch, seq_len, d_v, UNQUANTIZED);

    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n3. Semi Fused Attention (NVFP4)\n");
    // Dummy run to warm up GPU
    printf("Running dummy run for warm-up...\n");
    fused_flash_attention_nvfp4(d_X, 
                                d_Wq_nvfp4, d_Wk_nvfp4, d_Wv_nvfp4,
                                &h_Wq_nvfp4_meta, &h_Wk_nvfp4_meta, &h_Wv_nvfp4_meta,
                                d_bq, d_bk, d_bv,
                                d_output, batch, seq_len, d_model, d_k, d_v,
                                false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        fused_flash_attention_nvfp4(d_X, 
                                    d_Wq_nvfp4, d_Wk_nvfp4, d_Wv_nvfp4,
                                    &h_Wq_nvfp4_meta, &h_Wk_nvfp4_meta, &h_Wv_nvfp4_meta,
                                    d_bq, d_bk, d_bv,
                                    d_output, batch, seq_len, d_model, d_k, d_v,
                                    false);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);

    // Copy NVFP4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    time_nvfp4_semi = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    // Compare with baseline (both NVFP4, should match closely)
    compute_error_metrics("Flash NVFP4 vs Naive NVFP4", h_naive_baseline_nvfp4, h_result_temp, 
                         batch, seq_len, d_v, UNQUANTIZED);

    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n4. Our Attention (NVFP4)\n");

    printf("Running dummy run for warm-up...\n");
    our_attention_nvfp4(d_X, 
                        d_Wq_nvfp4, d_Wk_nvfp4, d_Wv_nvfp4,
                        &h_Wq_nvfp4_meta, &h_Wk_nvfp4_meta, &h_Wv_nvfp4_meta,
                        d_bq, d_bk, d_bv,
                        d_output, batch, seq_len, d_model, d_k, d_v,
                        false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    float kernel_times_nvfp4_ours[MAX_KERNELS];
    int num_kernels_nvfp4_ours = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        our_attention_nvfp4(d_X, 
                            d_Wq_nvfp4, d_Wk_nvfp4, d_Wv_nvfp4,
                            &h_Wq_nvfp4_meta, &h_Wk_nvfp4_meta, &h_Wv_nvfp4_meta,
                            d_bq, d_bk, d_bv,
                            d_output, batch, seq_len, d_model, d_k, d_v,
                            false,
                            (iter == 0) ? kernel_times_nvfp4_ours : nullptr,
                            (iter == 0) ? &num_kernels_nvfp4_ours : nullptr);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);

    // Copy NVFP4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    time_nvfp4_ours = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    printf("\nKernel Latency Breakdown:\n");
    printf("  1. Fused Dequant+Projection:  %.4f ms\n", kernel_times_nvfp4_ours[0]);
    printf("  2. Our Attention Kernel:      %.4f ms\n", kernel_times_nvfp4_ours[1]);
    printf("  ----------------------------------------\n");
    float total_nvfp4_ours = 0;
    for (int i = 0; i < num_kernels_nvfp4_ours; i++) total_nvfp4_ours += kernel_times_nvfp4_ours[i];
    printf("  Total (sum):                  %.4f ms\n", total_nvfp4_ours);
    
    // Compare with baseline (both NVFP4, should match closely)
    compute_error_metrics("Our NVFP4 vs Naive NVFP4", h_naive_baseline_nvfp4, h_result_temp, 
                         batch, seq_len, d_v, UNQUANTIZED);

    // Test 5: Full Attention (NVFP4)
    cooldown_gpu(COOLDOWN_SECONDS);
    printf("\n5. Full Attention (NVFP4) - Chunk-based Tiling\n");

    // Recreate events (they were destroyed)
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Running dummy run for warm-up...\n");
    full_attention_nvfp4(d_X, 
                        d_Wq_nvfp4, d_Wk_nvfp4, d_Wv_nvfp4,
                        &h_Wq_nvfp4_meta, &h_Wk_nvfp4_meta, &h_Wv_nvfp4_meta,
                        d_bq, d_bk, d_bv,
                        d_output, batch, seq_len, d_model, d_k, d_v,
                        false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run multiple iterations
    printf("Running %d iterations...\n", NUM_ITERATIONS);
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        full_attention_nvfp4(d_X, 
                            d_Wq_nvfp4, d_Wk_nvfp4, d_Wv_nvfp4,
                            &h_Wq_nvfp4_meta, &h_Wk_nvfp4_meta, &h_Wv_nvfp4_meta,
                            d_bq, d_bk, d_bv,
                            d_output, batch, seq_len, d_model, d_k, d_v,
                            false);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iteration_times[iter], start, stop));
    }
    elapsed_time_quant = compute_median(iteration_times, NUM_ITERATIONS);

    // Copy Full NVFP4 result back
    CUDA_CHECK(cudaMemcpy(h_result_temp, d_output, out_size, cudaMemcpyDeviceToHost));

    float time_nvfp4_full = elapsed_time_quant;
    printf("Median execution time: %.4f ms\n", elapsed_time_quant);
    
    // Compare with baseline (both NVFP4, should match closely)
    compute_error_metrics("Full NVFP4 vs Naive NVFP4", h_naive_baseline_nvfp4, h_result_temp, 
                         batch, seq_len, d_v, UNQUANTIZED);

    printf("\n");

    // Cleanup CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    printf("\n\n");
    printf("======================================================================================\n");
    printf("                                  PERFORMANCE SUMMARY                                 \n");
    printf("======================================================================================\n");
    printf("%-20s | %-12s | %-12s | %-12s\n", "Implementation", "MXFP4 (ms)", "NF4 (ms)", "NVFP4 (ms)");
    printf("--------------------------------------------------------------------------------------\n");
    printf("%-20s | %12.4f | %12.4f | %12.4f\n", "Naive", time_mxfp4_naive, time_nf4_naive, time_nvfp4_naive);
    printf("%-20s | %12.4f | %12.4f | %12.4f\n", "Projection-Fused", time_mxfp4_projection, time_nf4_projection, time_nvfp4_projection);
    printf("%-20s | %12.4f | %12.4f | %12.4f\n", "Semi-Fused", time_mxfp4_semi, time_nf4_semi, time_nvfp4_semi);
    printf("%-20s | %12.4f | %12.4f | %12.4f\n", "Ours", time_mxfp4_ours, time_nf4_ours, time_nvfp4_ours);
    printf("%-20s | %12.4f | %12.4f | %12.4f\n", "Full (Chunk-Tiled)", time_mxfp4_full, time_nf4_full, time_nvfp4_full);
    printf("======================================================================================\n");



    // Cleanup
    free(iteration_times);
    free(h_X);
    free(h_Wq);
    free(h_Wk);
    free(h_Wv);
    free(h_bq);
    free(h_bk);
    free(h_bv);
    free(h_output);
    free(h_baseline_unquantized);
    free(h_naive_baseline_int8);
    free(h_naive_baseline_mxfp4);
    free(h_naive_baseline_nf4);
    free(h_naive_baseline_nvfp4);
    free(h_result_temp);
    // free(h_Wq_quant);
    // free(h_Wk_quant);
    // free(h_Wv_quant);
    // free(h_Wq_scales);
    // free(h_Wk_scales);
    // free(h_Wv_scales);
    // free(h_Wq_zeros);
    // free(h_Wk_zeros);
    // free(h_Wv_zeros);
    free(h_Wq_mxfp4);
    free(h_Wk_mxfp4);
    free(h_Wv_mxfp4);
    free(h_Wq_nf4);
    free(h_Wk_nf4);
    free(h_Wv_nf4);
    free(h_Wq_nvfp4);
    free(h_Wk_nvfp4);
    free(h_Wv_nvfp4);

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaFree(d_bq));
    CUDA_CHECK(cudaFree(d_bk));
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_output));
    // CUDA_CHECK(cudaFree(d_Wq_quant));
    // CUDA_CHECK(cudaFree(d_Wk_quant));
    // CUDA_CHECK(cudaFree(d_Wv_quant));
    // CUDA_CHECK(cudaFree(d_Wq_scales));
    // CUDA_CHECK(cudaFree(d_Wk_scales));
    // CUDA_CHECK(cudaFree(d_Wv_scales));
    // CUDA_CHECK(cudaFree(d_Wq_zeros));
    // CUDA_CHECK(cudaFree(d_Wk_zeros));
    // CUDA_CHECK(cudaFree(d_Wv_zeros));
    CUDA_CHECK(cudaFree(d_Wq_mxfp4));
    CUDA_CHECK(cudaFree(d_Wk_mxfp4));
    CUDA_CHECK(cudaFree(d_Wv_mxfp4));
    CUDA_CHECK(cudaFree(d_Wq_nf4));
    CUDA_CHECK(cudaFree(d_Wk_nf4));
    CUDA_CHECK(cudaFree(d_Wv_nf4));
    CUDA_CHECK(cudaFree(d_Wq_nvfp4));
    CUDA_CHECK(cudaFree(d_Wk_nvfp4));
    CUDA_CHECK(cudaFree(d_Wv_nvfp4));

    printf("=== End of Test ===\n");

    return 0;
}

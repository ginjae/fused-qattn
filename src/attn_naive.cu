#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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

// Kernel 1: Q·Kᵀ MatMul
// Q: [batch, seq_len, d_k]
// K: [batch, seq_len, d_k]
// Output: [batch, seq_len, seq_len]
__global__ void qk_matmul_kernel(
    const float* Q,
    const float* K,
    float* QK,
    int batch,
    int seq_len,
    int d_k
) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && row < seq_len && col < seq_len) {
        float sum = 0.0f;

        // Compute dot product of Q[b, row, :] and K[b, col, :]
        for (int k = 0; k < d_k; k++) {
            int q_idx = b * seq_len * d_k + row * d_k + k;
            int k_idx = b * seq_len * d_k + col * d_k + k;
            sum += Q[q_idx] * K[k_idx];
        }

        int out_idx = b * seq_len * seq_len + row * seq_len + col;
        QK[out_idx] = sum;
    }
}

// Kernel 2: Scaling + Masking
// QK: [batch, seq_len, seq_len]
// scale_factor: 1/sqrt(d_k)
// causal_mask: true for causal attention
__global__ void scale_mask_kernel(
    float* QK,
    int batch,
    int seq_len,
    float scale_factor,
    bool causal_mask
) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && row < seq_len && col < seq_len) {
        int idx = b * seq_len * seq_len + row * seq_len + col;

        // Scale
        QK[idx] *= scale_factor;

        // Apply causal mask (mask future positions)
        if (causal_mask && col > row) {
            QK[idx] = -INFINITY;
        }
    }
}

// Kernel 3: Softmax (row-wise)
// QK: [batch, seq_len, seq_len]
// Output: [batch, seq_len, seq_len]
__global__ void softmax_kernel(
    const float* QK,
    float* A,
    int batch,
    int seq_len
) {
    int b = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && row < seq_len) {
        int base_idx = b * seq_len * seq_len + row * seq_len;

        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < seq_len; i++) {
            max_val = fmaxf(max_val, QK[base_idx + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float exp_val = expf(QK[base_idx + i] - max_val);
            A[base_idx + i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for (int i = 0; i < seq_len; i++) {
            A[base_idx + i] /= sum;
        }
    }
}

// Kernel 4: Attention Score Application (A·V)
// A: [batch, seq_len, seq_len] - attention scores
// V: [batch, seq_len, d_v]
// Output: [batch, seq_len, d_v]
__global__ void av_matmul_kernel(
    const float* A,
    const float* V,
    float* output,
    int batch,
    int seq_len,
    int d_v
) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && row < seq_len && col < d_v) {
        float sum = 0.0f;

        // Compute dot product of A[b, row, :] and V[b, :, col]
        for (int k = 0; k < seq_len; k++) {
            int a_idx = b * seq_len * seq_len + row * seq_len + k;
            int v_idx = b * seq_len * d_v + k * d_v + col;
            sum += A[a_idx] * V[v_idx];
        }

        int out_idx = b * seq_len * d_v + row * d_v + col;
        output[out_idx] = sum;
    }
}

// Host function to perform complete naive attention with weight matrices
void naive_attention(
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
    // Allocate temporary buffers for Q, K, V, QK, A
    float *d_Q, *d_K, *d_V, *d_QK, *d_A;
    CUDA_CHECK(cudaMalloc(&d_Q, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, batch * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, batch * seq_len * d_v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_QK, batch * seq_len * seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A, batch * seq_len * seq_len * sizeof(float)));

    dim3 block_proj(16, 16);

    // Step 0-1: Compute Q = X·Wq + bq
    dim3 grid_q((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_kernel<<<grid_q, block_proj>>>(d_X, d_Wq, d_bq, d_Q, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-2: Compute K = X·Wk + bk
    dim3 grid_k((d_k + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_kernel<<<grid_k, block_proj>>>(d_X, d_Wk, d_bk, d_K, batch, seq_len, d_model, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 0-3: Compute V = X·Wv + bv
    dim3 grid_v((d_v + block_proj.x - 1) / block_proj.x,
                (seq_len + block_proj.y - 1) / block_proj.y,
                batch);
    linear_projection_kernel<<<grid_v, block_proj>>>(d_X, d_Wv, d_bv, d_V, batch, seq_len, d_model, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Step 1: Q·Kᵀ MatMul
    dim3 block1(16, 16);
    dim3 grid1((seq_len + block1.x - 1) / block1.x,
               (seq_len + block1.y - 1) / block1.y,
               batch);
    qk_matmul_kernel<<<grid1, block1>>>(d_Q, d_K, d_QK, batch, seq_len, d_k);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Scaling + Masking
    float scale_factor = 1.0f / sqrtf((float)d_k);
    scale_mask_kernel<<<grid1, block1>>>(d_QK, batch, seq_len, scale_factor, causal_mask);
    CUDA_CHECK(cudaGetLastError());

    // Step 3: Softmax
    dim3 block2(256);
    dim3 grid2((seq_len + block2.x - 1) / block2.x, batch);
    softmax_kernel<<<grid2, block2>>>(d_QK, d_A, batch, seq_len);
    CUDA_CHECK(cudaGetLastError());

    // Step 4: A·V MatMul
    dim3 block3(16, 16);
    dim3 grid3((d_v + block3.x - 1) / block3.x,
               (seq_len + block3.y - 1) / block3.y,
               batch);
    av_matmul_kernel<<<grid3, block3>>>(d_A, d_V, d_output, batch, seq_len, d_v);
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_QK));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaDeviceSynchronize());
}


// Simple test function
int main() {
    // Test parameters
    int batch = 1;
    int seq_len = 4;
    int d_model = 64;
    int d_k = 8;
    int d_v = 8;

    printf("=== Starting Test: Naive Attention ===\n");

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

    // Run naive attention with weights
    printf("Running naive attention with weight matrices...\n");
    naive_attention(d_X, d_Wq, d_Wk, d_Wv, d_bq, d_bk, d_bv, 
                                  d_output, batch, seq_len, d_model, d_k, d_v, false);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost));

    // Print results
    printf("\nOutput:\n");
    for (int i = 0; i < seq_len; i++) {
        printf("Position %d: [", i);
        for (int j = 0; j < d_v; j++) {
            int idx = i * d_v + j;
            printf("%.6f", h_output[idx]);
            if (j < d_v - 1) printf(", ");
        }
        printf("]\n");
    }
    printf("\n");


    // Cleanup
    free(h_X);
    free(h_Wq);
    free(h_Wk);
    free(h_Wv);
    free(h_bq);
    free(h_bk);
    free(h_bv);
    free(h_output);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaFree(d_bq));
    CUDA_CHECK(cudaFree(d_bk));
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_output));

    printf("=== End of Test ===\n");

    return 0;
}

#ifndef ATTN_FULL_CUH
#define ATTN_FULL_CUH

#include <stdint.h>
#include "quantization_utils.cuh"

// Full attention functions - completely fused kernels
// These perform dequantization + QKV projection + attention in a single kernel
// without storing intermediate Q, K, V buffers in global memory

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
    bool causal_mask = false
);

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
    bool causal_mask = false
);

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
    bool causal_mask = false
);

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
    bool causal_mask = false
);

#endif // ATTN_FULL_CUH


#ifndef ATTN_FLASH_CUH
#define ATTN_FLASH_CUH

#include <stdint.h>

// Flash attention functions
void flash_attention(
    const float* X,
    const float* Wq,
    const float* Wk,
    const float* Wv,
    const float* bq,
    const float* bk,
    const float* bv,
    float* output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    bool verbose
);

void flash_attention_quantized_blockwise(
    const float* X,
    const int8_t* Wq_quantized,
    const int8_t* Wk_quantized,
    const int8_t* Wv_quantized,
    const float* Wq_scales,
    const float* Wk_scales,
    const float* Wv_scales,
    const int8_t* Wq_zeros,
    const int8_t* Wk_zeros,
    const int8_t* Wv_zeros,
    const float* bq,
    const float* bk,
    const float* bv,
    float* output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    int block_size,
    bool verbose
);

void flash_attention_mxfp4(
    const float* X,
    const void* Wq_mxfp4,
    const void* Wk_mxfp4,
    const void* Wv_mxfp4,
    const float* bq,
    const float* bk,
    const float* bv,
    float* output,
    int batch,
    int seq_len,
    int d_model,
    int d_k,
    int d_v,
    bool verbose
);

#endif // ATTN_FLASH_CUH

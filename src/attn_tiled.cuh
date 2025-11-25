#ifndef ATTN_TILED_CUH
#define ATTN_TILED_CUH

#include <stdint.h>

// Tiled attention functions
void tiled_attention(
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

void tiled_attention_quantized(
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

#endif // ATTN_TILED_CUH

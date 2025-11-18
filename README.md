# Fused CUDA Kernels for Quantized LLM Attention

This repository explores the design and implementation of a fused CUDA kernel tailored for **quantized Large Language Models (LLMs)**, focusing on **attention layers** that operate on **block-wise quantized weights**, (e.g., NF4, MXFP4, NVFP4).  
The core objective is to integrate **dequantization** and **attention operations** (matmul, scaling, masking, softmax, ...) into a single I/O-aware kernel to substantially reduce memory traffic and improve inference throughput.  


## 🚀 Overview

Modern LLMs rely heavily on the attention mechanism, which becomes **memory-bound** due to repeated global memory accesses across multiple kernels. Quantized models alleviate storage cost but introduce **additional overhead for dequantization**, typically executed as a standalone kernel before matrix multiplication.

This project tackles this bottleneck by fusing:

1. **Block-wise Dequantization**  
2. **Q·Kᵀ MatMul with on-chip tiling**  
3. **Scaling + Masking**  
4. **Chunk-wise Softmax**  
5. **Attention Score Application (A·V)**  

all within a **single CUDA kernel**, inspired by FlashAttention-style I/O-aware design.


## 🤝 Contributors

[Jaemin Kim](mailto:jm611@unist.ac.kr) (20211061)

[Jiseung Jeong](mailto:wjdwltmd1151@unist.ac.kr) (20211301)

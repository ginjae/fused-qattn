# 🔥 Fused CUDA Kernels for Quantized LLM Attention

This repository explores the design and implementation of a fused CUDA kernel tailored for **quantized Large Language Models (LLMs)**, focusing on **attention layers** that operate on **block-wise quantized weights**, (e.g., NF4, MXFP4, NVFP4).  
The core objective is to integrate **dequantization** and **Q/K/V projection operations** into a single I/O-aware kernel, followed by a separate fused attention kernel, to substantially reduce memory traffic and improve inference throughput.  
> This project was implemented using CUDA 12.4.


## 🚀 Quick Start

### 📦 Extract Real GPT-2 Weights (Recommended)

For more realistic evaluation, you can use actual GPT-2 attention weights:

```bash
# Install required dependencies
pip install torch transformers numpy

# Extract weights from GPT-2 (layer 0, head 0 by default)
python extract_gpt2_weights.py --output weights

# Or specify different layer/head
python extract_gpt2_weights.py --model gpt2 --layer 0 --head 0 --output weights
```

This will create a `weights/` directory containing:
- `wq.npy`, `wk.npy`, `wv.npy` - Query, Key, Value projection matrices
- `bq.npy`, `bk.npy`, `bv.npy` - Corresponding biases
- `metadata.json` - Weight metadata

The evaluation code will automatically load these weights if available, otherwise it falls back to random initialization.

### ⚙️ Build and Run

```bash
make run_eval
```


## 📊 Overview

Naïve attention implementations in modern Large Language Models (LLMs) are fundamentally **memory-bound**, as it requires multiple passes over large activation and weight tensors across several independent CUDA kernels. When using **block-wise weight-only quantization** formats such as NF4, MXFP4, or NVFP4, an additional overhead is introduced by **dequantization**, which is typically executed as a standalone kernel. This increases global memory traffic and undermines the data locality necessary for high-throughput inference.

This project addresses these challenges by introducing a **two-kernel, quantization-aware fused attention pipeline**, inspired by the I/O-aware philosophy of FlashAttention but optimized for quantized LLMs. The objective is to reorganize computation so that intermediate results remain on-chip, minimizing redundant memory transfers.

### 🔢 Why Block-wise Quantization Requires Fused Dequantization

**Block-wise Quantization** offers lower quantization error compared to element-wise quantization, but it necessitates a dequantization step during inference. This process requires **metadata** such as per-block scale factors, zero-points, and **Look-Up Tables (LUT)** for formats like NF4 that map quantized indices to floating-point values. There are two approaches to handle this:

#### ❌ Offline Dequantization (Pre-conversion)
- Dequantized Q', K', V' tensors (fp32) reside in memory
- Original quantized tensors often cannot be freed immediately due to implementation constraints
- Effectively results in memory usage similar to an "fp32 model"
- **→ Minimal memory benefits**

#### ✅ Fused Dequantization (On-the-fly)
- No intermediate fp32 tensors stored in memory
- Dequantization results are consumed directly in registers
- **→ For FP32 → 4-bit quantization, weight memory usage is reduced by ~8x**
  - *(Actual: ~5.5x–7x reduction due to metadata overhead: scales, zero-points, and LUTs)*

Our implementation leverages **fused dequantization** to achieve substantial memory savings while maintaining computational efficiency.

### 🧱 Two-Kernel Architecture

Our design consists of two fused kernels:

### 1️⃣ First Kernel — Block Dequantization + Fused Q/K/V Projection
This kernel performs block-wise dequantization of the quantized weight matrices and immediately consumes the dequantized values within a tiled matmul to compute $[Q\,|\,K\,|\,V] = X[W_Q\,|\,W_K\,|\,W_V]$.
Without writing dequantized weights back to global memory, it significantly reduces memory I/O and preserves data locality.

### 2️⃣ Second Kernel — Flash-style Tiled Attention
This kernel implements the core attention pipeline in a fully fused manner.
The tiled execution reuses Q, K, and V efficiently across shared memory and registers, further minimizing global memory access.

This two-kernel architecture balances aggressive operator fusion with practical implementation constraints. Compared to conventional multi-kernel attention and quantization workflows, it **significantly lowers memory traffic**, enhances **on-chip data reuse**, and delivers **substantial inference throughput improvements** for quantized LLMs.


# 📋 Baselines

## 🐌 Naïve Sequential Attention

There are six kernel functions, each of which is for:

1. $W_Q, W_K, W_V$ de-quantization
2. Naïve matmul: $Q=XW_Q, K=XW_K, V=XW_V$
3. Naïve matmul: $QK^\top$
4. Scaling: $QK^\top\over{\sqrt{d_k}}$

    \+
    Causal masking (Optional)
5. $\text{softmax}({QK^\top\over{\sqrt{d_k}}})$
6. Naïve matmul: $\text{Attention}(Q, K, V) = \text{softmax}({QK^\top\over{\sqrt{d_k}}}) V$ 

## 🧩 Tiled Sequential Attention

There are six kernel functions, each of which is for:

1. $W_Q, W_K, W_V$ de-quantization
2. Tiled matmul: $Q=XW_Q, K=XW_K, V=XW_V$
3. Tiled matmul: $QK^\top$
4. Scaling: $QK^\top\over{\sqrt{d_k}}$

    \+
    Causal masking (Optional)
5. $\text{chunk-wise softmax}({QK^\top\over{\sqrt{d_k}}})$
6. Tiled matmul: $\text{Attention}(Q, K, V) = \text{chunk-wise softmax}({QK^\top\over{\sqrt{d_k}}}) V$

## ⚡ Tiled Fused Attention (Flash-style)

There are three kernel functions, each of which is for:

1. $W_Q, W_K, W_V$ de-quantization
2. Fused & Tiled matmul: $[Q|K|V]=X[W_Q|W_K|W_V]$
3. Tiled matmul: $QK^\top$

    \+
    Scaling: $QK^\top\over{\sqrt{d_k}}$

    \+
    Causal masking (Optional)

    \+
    $\text{chunk-wise softmax}({QK^\top\over{\sqrt{d_k}}})$

    \+
    Tiled matmul: $\text{Attention}(Q, K, V) = \text{chunk-wise softmax}({QK^\top\over{\sqrt{d_k}}}) V$


# 🎯 Quantization-Aware Tiled Fused Attention (Ours)

There are two kernel functions, each of which is for:

1. $W_Q, W_K, W_V$ de-quantization 

    \+
    Fused & Tiled matmul: $[Q|K|V]=X[W_Q|W_K|W_V]$
2. Tiled matmul: $QK^\top$

    \+
    Scaling: $QK^\top\over{\sqrt{d_k}}$

    \+
    Causal masking (Optional)

    \+
    $\text{chunk-wise softmax}({QK^\top\over{\sqrt{d_k}}})$

    \+
    Tiled matmul: $\text{Attention}(Q, K, V) = \text{chunk-wise softmax}({QK^\top\over{\sqrt{d_k}}}) V$


## 🤝 Contributors

[Jaemin Kim](mailto:jm611@unist.ac.kr) (20211061)

[Jiseung Jeong](mailto:wjdwltmd1151@unist.ac.kr) (20211301)

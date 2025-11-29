#!/usr/bin/env python3
"""
Extract GPT-2 attention weights for evaluation.
This script extracts the Wq, Wk, Wv weights and biases from a pre-trained GPT-2 model
and saves them in a format that can be loaded by the CUDA evaluation code.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel
import argparse
import os

def extract_gpt2_weights(model_name='gpt2', layer_idx=0, head_idx=0, output_dir='weights'):
    """
    Extract attention weights from GPT-2 model.
    
    Args:
        model_name: GPT-2 model variant ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        layer_idx: Which transformer layer to extract from (0-11 for gpt2)
        head_idx: Which attention head to extract (0-11 for gpt2)
        output_dir: Directory to save the extracted weights
    """
    print(f"Loading {model_name} model...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    # Get model config
    config = model.config
    n_layer = config.n_layer
    n_head = config.n_head
    n_embd = config.n_embd
    
    print(f"Model config: {n_layer} layers, {n_head} heads, {n_embd} embedding dim")
    
    if layer_idx >= n_layer:
        raise ValueError(f"layer_idx {layer_idx} >= n_layer {n_layer}")
    if head_idx >= n_head:
        raise ValueError(f"head_idx {head_idx} >= n_head {n_head}")
    
    # Calculate head dimension
    d_head = n_embd // n_head
    
    print(f"Extracting weights from layer {layer_idx}, head {head_idx}")
    print(f"Head dimension: {d_head}")
    
    # Get the attention layer
    attn_layer = model.transformer.h[layer_idx].attn
    
    # GPT-2 uses a single matrix for Q, K, V projection: c_attn
    # Shape: [n_embd, 3 * n_embd]
    c_attn_weight = attn_layer.c_attn.weight.data  # [n_embd, 3*n_embd]
    c_attn_bias = attn_layer.c_attn.bias.data      # [3*n_embd]
    
    # Split into Q, K, V
    # Note: PyTorch weights are transposed compared to typical notation
    qkv_weight = c_attn_weight.t()  # [3*n_embd, n_embd]
    q_weight = qkv_weight[:n_embd, :]           # [n_embd, n_embd]
    k_weight = qkv_weight[n_embd:2*n_embd, :]   # [n_embd, n_embd]
    v_weight = qkv_weight[2*n_embd:, :]         # [n_embd, n_embd]
    
    qkv_bias = c_attn_bias
    q_bias = qkv_bias[:n_embd]           # [n_embd]
    k_bias = qkv_bias[n_embd:2*n_embd]   # [n_embd]
    v_bias = qkv_bias[2*n_embd:]         # [n_embd]
    
    # Extract single head weights
    # Reshape to [n_head, d_head, n_embd] and extract the specific head
    q_weight_all_heads = q_weight.reshape(n_head, d_head, n_embd)  # [n_head, d_head, n_embd]
    k_weight_all_heads = k_weight.reshape(n_head, d_head, n_embd)
    v_weight_all_heads = v_weight.reshape(n_head, d_head, n_embd)
    
    q_bias_all_heads = q_bias.reshape(n_head, d_head)  # [n_head, d_head]
    k_bias_all_heads = k_bias.reshape(n_head, d_head)
    v_bias_all_heads = v_bias.reshape(n_head, d_head)
    
    # Extract specific head
    wq_head = q_weight_all_heads[head_idx]  # [d_head, n_embd]
    wk_head = k_weight_all_heads[head_idx]
    wv_head = v_weight_all_heads[head_idx]
    
    bq_head = q_bias_all_heads[head_idx]  # [d_head]
    bk_head = k_bias_all_heads[head_idx]
    bv_head = v_bias_all_heads[head_idx]
    
    # Transpose to match expected format: [n_embd, d_head]
    wq_head = wq_head.t()  # [n_embd, d_head]
    wk_head = wk_head.t()
    wv_head = wv_head.t()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy arrays (binary format for efficient loading in C++)
    np.save(os.path.join(output_dir, 'wq.npy'), wq_head.cpu().numpy().astype(np.float32))
    np.save(os.path.join(output_dir, 'wk.npy'), wk_head.cpu().numpy().astype(np.float32))
    np.save(os.path.join(output_dir, 'wv.npy'), wv_head.cpu().numpy().astype(np.float32))
    np.save(os.path.join(output_dir, 'bq.npy'), bq_head.cpu().numpy().astype(np.float32))
    np.save(os.path.join(output_dir, 'bk.npy'), bk_head.cpu().numpy().astype(np.float32))
    np.save(os.path.join(output_dir, 'bv.npy'), bv_head.cpu().numpy().astype(np.float32))
    
    # Also save a metadata file
    metadata = {
        'model_name': model_name,
        'layer_idx': layer_idx,
        'head_idx': head_idx,
        'n_embd': n_embd,
        'd_head': d_head,
        'n_head': n_head,
        'n_layer': n_layer,
        'wq_shape': list(wq_head.shape),
        'wk_shape': list(wk_head.shape),
        'wv_shape': list(wv_head.shape),
        'bq_shape': list(bq_head.shape),
        'bk_shape': list(bk_head.shape),
        'bv_shape': list(bv_head.shape),
    }
    
    import json
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nWeights saved to {output_dir}/")
    print(f"  Wq shape: {wq_head.shape}")
    print(f"  Wk shape: {wk_head.shape}")
    print(f"  Wv shape: {wv_head.shape}")
    print(f"  bq shape: {bq_head.shape}")
    print(f"  bk shape: {bk_head.shape}")
    print(f"  bv shape: {bv_head.shape}")
    
    return metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract GPT-2 attention weights')
    parser.add_argument('--model', type=str, default='gpt2', 
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='GPT-2 model variant')
    parser.add_argument('--layer', type=int, default=0, 
                        help='Layer index to extract from')
    parser.add_argument('--head', type=int, default=0, 
                        help='Attention head index to extract')
    parser.add_argument('--output', type=str, default='weights',
                        help='Output directory for weights')
    
    args = parser.parse_args()
    
    extract_gpt2_weights(
        model_name=args.model,
        layer_idx=args.layer,
        head_idx=args.head,
        output_dir=args.output
    )

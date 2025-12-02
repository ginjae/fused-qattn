#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

def visualize_latency_breakdown(csv_path):
    """Visualize kernel latency breakdown from CSV file"""
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Extract GPU name and timestamp from filename
    filename = Path(csv_path).stem
    parts = filename.split('_')
    if len(parts) >= 4:
        gpu_name = ' '.join(parts[3:-2]).replace('_', ' ')
    else:
        gpu_name = "Unknown GPU"
    
    # First, generate performance comparison graph
    visualize_performance_from_latency(df, csv_path, gpu_name)
    
    # Get unique quantization types and implementations
    quant_types = df['Quantization'].unique()
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create subplots for each quantization type
    n_quants = len(quant_types)
    fig, axes = plt.subplots(1, n_quants, figsize=(6*n_quants, 6))
    if n_quants == 1:
        axes = [axes]
    
    fig.suptitle(f'Kernel Latency Breakdown - {gpu_name}', 
                 fontsize=16, fontweight='bold')
    
    for idx, quant_type in enumerate(quant_types):
        ax = axes[idx]
        
        # Filter data for this quantization type
        quant_df = df[df['Quantization'] == quant_type].copy()
        
        # Get unique implementations
        implementations = quant_df['Implementation'].unique()
        
        # Normalize kernel names for consistent grouping
        def normalize_kernel_name(kernel_name):
            kernel_lower = kernel_name.lower().replace(' ', '_')
            # Normalize attention kernel names
            if 'attention' in kernel_lower or 'flash' in kernel_lower:
                return 'FlashAttention'
            # Normalize other common variations
            kernel_name = kernel_name.replace(' ', '_')
            return kernel_name
        
        # Prepare data for stacked bar chart
        impl_data = {}
        for impl in implementations:
            impl_df = quant_df[quant_df['Implementation'] == impl]
            # Normalize kernel names
            normalized_kernels = [normalize_kernel_name(k) for k in impl_df['Kernel'].values]
            impl_data[impl] = {
                'kernels': np.array(normalized_kernels),
                'times': impl_df['Time_ms'].values
            }
        
        # Create stacked bar chart
        x_pos = np.arange(len(implementations))
        
        # Assign colors based on kernel category and Q/K/V
        def get_kernel_color(kernel_name):
            kernel_lower = kernel_name.lower()
            
            # Check if it's a fused kernel
            is_fused = kernel_name.startswith('Fused_')
            
            # Check for Q/K/V suffix
            qkv_suffix = None
            if '_q' in kernel_lower or 'wq' in kernel_lower:
                qkv_suffix = 'q'
            elif '_k' in kernel_lower or 'wk' in kernel_lower:
                qkv_suffix = 'k'
            elif '_v' in kernel_lower or 'wv' in kernel_lower:
                qkv_suffix = 'v'
            
            # Determine category and get base color
            if 'dequant' in kernel_lower:
                if is_fused and kernel_lower.startswith('fused_dequant'):
                    return '#CC00CC'  # Purple for fused dequant+projection
                # Red tones for dequantization
                elif is_fused:
                    return '#CC0000'  # Dark red for fused
                elif qkv_suffix == 'q':
                    return '#FF6B6B'  # Bright red
                elif qkv_suffix == 'k':
                    return '#FF8E8E'  # Medium red
                elif qkv_suffix == 'v':
                    return '#FFB1B1'  # Light red
                else:
                    return '#FF6B6B'  # Default red
            elif 'projection' in kernel_lower or 'linear' in kernel_lower:
                # Teal tones for projection
                if is_fused:
                    return '#2A9D8F'  # Dark teal for fused
                elif qkv_suffix == 'q':
                    return '#4ECDC4'  # Bright teal
                elif qkv_suffix == 'k':
                    return '#6FD9D0'  # Medium teal
                elif qkv_suffix == 'v':
                    return '#90E5DC'  # Light teal
                else:
                    return '#4ECDC4'  # Default teal
            elif 'attention' in kernel_lower or 'flash' in kernel_lower:
                return '#FFA726'  # Orange
            else:
                return '#A8A8A8'  # Gray
        
        # Plot stacked bars - each implementation has its own stack in original order
        all_patches = []
        all_labels = []
        
        for i, impl in enumerate(implementations):
            kernels = impl_data[impl]['kernels']
            times = impl_data[impl]['times']
            bottom = 0
            
            for kernel, time in zip(kernels, times):
                color = get_kernel_color(kernel)
                bar = ax.bar(x_pos[i], time, bottom=bottom, 
                           color=color, width=0.6, edgecolor='white', linewidth=0.5)
                
                # Store for legend (avoid duplicates)
                if kernel not in all_labels:
                    all_patches.append(bar[0])
                    all_labels.append(kernel)
                
                # Add time labels on bars (only if significant)
                if time > 0.005:  # Only show if > 0.005ms
                    y_pos = bottom + time / 2
                    ax.text(x_pos[i], y_pos,
                           f'{time:.3f}',
                           ha='center', va='center', fontsize=7, fontweight='bold')
                
                bottom += time
            
            # Add total time on top of each bar
            ax.text(x_pos[i], bottom, f'{bottom:.3f}ms',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Flash Attention Implementation', fontsize=11, fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'{quant_type}', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        # Replace underscores with spaces in x-axis labels
        ax.set_xticklabels([impl.replace('_', ' ') for impl in implementations], rotation=15, ha='right')
        # Replace underscores with spaces in legend labels - only show on rightmost subplot
        if idx == n_quants - 1:
            all_labels_display = [label.replace('_', ' ') for label in all_labels]
            ax.legend(all_patches, all_labels_display, loc='upper left', fontsize=8, bbox_to_anchor=(1.02, 1))
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = csv_path.replace('.csv', '.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show plot
    plt.show()

def visualize_performance_from_latency(df, csv_path, gpu_name):
    """Generate performance comparison graph from latency breakdown data"""
    
    # Calculate total time per implementation and quantization type
    perf_data = df.groupby(['Quantization', 'Implementation'])['Time_ms'].sum().reset_index()
    
    # Pivot to get implementations as rows and quantization types as columns
    perf_pivot = perf_data.pivot(index='Implementation', columns='Quantization', values='Time_ms')
    
    # Preserve original order from CSV
    implementations = df['Implementation'].unique()
    quant_types = df['Quantization'].unique()
    
    # Reindex to maintain original order
    perf_pivot = perf_pivot.reindex(index=implementations, columns=quant_types)
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(f'Performance Comparison - {gpu_name}', 
                 fontsize=16, fontweight='bold')
    
    width = 0.8 / len(implementations)
    
    # Get baseline (first implementation) times for speedup calculation
    baseline_times = perf_pivot.iloc[0].values
    
    # Grouped bar chart
    for i, impl in enumerate(implementations):
        values = perf_pivot.loc[impl].values
        x_pos = np.arange(len(quant_types)) + i * width
        # Add " Flash Attention" to legend label
        label_text = f"{impl.replace('_', ' ')} Flash Attention" if impl != "Ours" else impl
        bars = ax.bar(x_pos, values, width, label=label_text, color=colors[i])
        
        # Add value labels and speedup on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                height = bar.get_height()
                
                # Show speedup (except for first baseline)
                if i > 0 and not np.isnan(baseline_times[j]):
                    speedup = baseline_times[j] / val
                    # label_text = f'{speedup:.2f}×\n{val:.3f}ms'
                    label_text = f'{val:.3f}ms'
                    text_color = 'black'
                    # text_color = 'darkgreen' if speedup > 1 else 'darkred'
                else:
                    # Just show execution time for baseline
                    label_text = f'{val:.3f}ms'
                    text_color = 'black'
                
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        label_text,
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        linespacing=0.8, color=text_color)
    
    ax.set_xlabel('Quantization Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Total Execution Time by Quantization Type', fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(quant_types)) + width * (len(implementations) - 1) / 2)
    ax.set_xticklabels(quant_types)
    ax.grid(axis='y', alpha=0.3)
    
    # Increase y-axis limit to make room for legend
    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.15)
    
    legend = ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('gray')
    legend.set_zorder(100)
    
    plt.tight_layout()
    
    # Save figure
    output_path = csv_path.replace('latency_breakdown_', 'performance_').replace('.csv', '.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance graph saved to: {output_path}")
    
    plt.show()

def visualize_performance(csv_path):
    """Visualize performance results from CSV file"""
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Extract performance data (first 4 rows)
    perf_df = df.iloc[:4].copy()
    
    # Replace 'N/A' with NaN for proper handling
    perf_df = perf_df.replace('N/A', np.nan)
    
    # Convert to numeric
    for col in ['No_Quant_ms', 'MXFP4_ms', 'NF4_ms', 'NVFP4_ms']:
        perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce')
    
    # Extract GPU name and timestamp from filename
    filename = Path(csv_path).stem
    parts = filename.split('_')
    if len(parts) >= 4:
        gpu_name = ' '.join(parts[1:-2]).replace('_', ' ')
    else:
        gpu_name = "Unknown GPU"
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(f'Performance Benchmark Results - {gpu_name}', 
                 fontsize=16, fontweight='bold')
    
    implementations = perf_df['Implementation'].values
    quant_types = ['No_Quant_ms', 'MXFP4_ms', 'NF4_ms', 'NVFP4_ms']
    quant_labels = ['No Quant', 'MXFP4', 'NF4', 'NVFP4']
    width = 0.18
    
    # Get baseline (Naive) times for speedup calculation
    naive_times = perf_df.iloc[0, 1:].values
    
    # Grouped bar chart: Per quantization type
    for i, impl in enumerate(implementations):
        values = [perf_df.loc[i, col] for col in quant_types]
        x_pos = np.arange(len(quant_types)) + i * 0.2
        bars = ax.bar(x_pos, values, width, label=impl, color=colors[i])
        
        # Add value labels and speedup on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                height = bar.get_height()
                
                # Show speedup (except for Naive baseline)
                if i > 0 and not np.isnan(naive_times[j]):
                    speedup = naive_times[j] / val
                    
                    label_text = f'{val:.3f}ms'
                    # label_text = f'({speedup:.2f}×)\n{val:.3f}ms'
                    text_color = 'black'
                else:
                    # Just show execution time for Naive
                    label_text = f'{val:.3f}ms'
                    text_color = 'black'
                
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        label_text,
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        linespacing=0.9, color=text_color)
    
    ax.set_xlabel('Quantization Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Quantization Type', fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(quant_types)) + 0.3)
    ax.set_xticklabels(quant_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = csv_path.replace('.csv', '.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Find the most recent CSV file in results directory
        results_dir = Path(__file__).parent / "results"
        csv_files = list(results_dir.glob("*.csv"))
        if not csv_files:
            print("Error: No CSV files found in results/ directory")
            sys.exit(1)
        csv_path = str(max(csv_files, key=os.path.getmtime))
        print(f"Using most recent CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Determine which visualization to use based on filename
    if "latency_breakdown" in csv_path:
        visualize_latency_breakdown(csv_path)
    else:
        visualize_performance(csv_path)

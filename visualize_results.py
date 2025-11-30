#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

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
        csv_files = list(results_dir.glob("performance_*.csv"))
        if not csv_files:
            print("Error: No CSV files found in results/ directory")
            sys.exit(1)
        csv_path = str(max(csv_files, key=os.path.getmtime))
        print(f"Using most recent CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    visualize_performance(csv_path)

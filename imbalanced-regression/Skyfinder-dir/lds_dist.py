import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
import os

# --- LDS Kernel Helper (extracted from your project's logic) ---
def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [np.exp(-x**2 / (2 * sigma**2)) for x in range(-half_ks, half_ks + 1)]
    elif kernel == 'triang':
        base_kernel = [1 - abs(x) / (half_ks + 1) for x in range(-half_ks, half_ks + 1)]
    else: # laplace
        base_kernel = [np.exp(-abs(x) / sigma) for x in range(-half_ks, half_ks + 1)]
    kernel_window = np.array(base_kernel) / sum(base_kernel)
    return kernel_window

def save_smoothed_distribution(csv_path, output_png, lds_ks=9, lds_sigma=1, lds_kernel='gaussian'):
    # 1. Load Data
    df = pd.read_csv(csv_path)
    df_train = df[df['split'] == 'train']
    labels = df_train['label'].values
    
    # 2. Prepare Raw Distribution (0-80 as per your setup)
    max_target = 81
    value_dict = {x: 0 for x in range(max_target)}
    for label in labels:
        value_dict[min(max_target - 1, int(label))] += 1
    
    raw_counts = np.asarray([v for _, v in value_dict.items()])
    
    # 3. Apply LDS Smoothing
    lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
    smoothed_counts = convolve1d(raw_counts, weights=lds_kernel_window, mode='constant')
    
    # 4. Plot and Save
    plt.figure(figsize=(12, 6))
    plt.bar(range(max_target), raw_counts, alpha=0.3, color='steelblue', label='Raw Empirical Distribution')
    plt.plot(range(max_target), smoothed_counts, color='crimson', linewidth=2.5, label='LDS Effective Distribution')
    
    plt.title(f'LDS Smoothing (Kernel: {lds_kernel}, KS: {lds_ks}, Sigma: {lds_sigma})')
    plt.xlabel('Target Value (Age)')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(axis='y', alpha=0.2)
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_png}")
    plt.close()

if __name__ == "__main__":
    # Change these settings as needed

    lds_ks = 5
    lds_sigma = 2

    CSV_PATH = "./data/skyfinder_30_balanced.csv" 
    OUTPUT_NAME = f"smoothed_dist_ks{lds_ks}_sigma{lds_sigma}.png"
    
    save_smoothed_distribution(
        csv_path=CSV_PATH, 
        output_png=OUTPUT_NAME,
        lds_ks=lds_ks,      # Kernel size: defines neighborhood width 
        lds_sigma=lds_sigma,   # Sigma: defines smoothing spread 
        lds_kernel='gaussian' # Options: 'gaussian', 'triang', 'laplace' [cite: 362, 523]
    )
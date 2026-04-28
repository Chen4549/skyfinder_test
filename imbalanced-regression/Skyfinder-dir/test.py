import pandas as pd
import numpy as np

def calculate_dataset_stats(csv_path, many_thr=100, low_thr=20, max_target=81):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # We focus specifically on the 'train' split as that defines the imbalance the model learns
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        print(f"Total samples in training set: {len(train_df)}")
    else:
        train_df = df
        print(f"Total samples in dataset: {len(train_df)}")

    labels = train_df['label'].values
    
    # Initialize bin counts (0 to 80)
    bin_counts = np.zeros(max_target)
    for label in labels:
        bin_idx = int(min(max_target - 1, label))
        bin_counts[bin_idx] += 1

    # Categorize bins
    many_shot_bins = np.where(bin_counts > many_thr)[0]
    low_shot_bins = np.where((bin_counts < low_thr) & (bin_counts > 0))[0]
    median_shot_bins = np.where((bin_counts >= low_thr) & (bin_counts <= many_thr))[0]
    zero_shot_bins = np.where(bin_counts == 0)[0]

    # Print Statistics
    print("-" * 30)
    print(f"Statistics (Thresholds: Many > {many_thr}, Low < {low_thr})")
    print("-" * 30)
    print(f"Many-shot regions:   {len(many_shot_bins)} bins")
    print(f"Median-shot regions: {len(median_shot_bins)} bins")
    print(f"Low-shot regions:    {len(low_shot_bins)} bins")
    print(f"Zero-shot regions:   {len(zero_shot_bins)} bins")
    print("-" * 30)
    print(f"Max samples in a bin: {int(np.max(bin_counts))}")
    print(f"Min samples in a bin: {int(np.min(bin_counts[bin_counts > 0]))}")
    print(f"Average bin density:  {np.mean(bin_counts[bin_counts > 0]):.2f}")
    
    return bin_counts

if __name__ == "__main__":
    # Update with your file path
    CSV_FILE = "./data/skyfinder_30.csv" 
    
    # For 90,000 samples, you might want to test higher thresholds 
    # like many_thr=500, low_thr=50
    counts = calculate_dataset_stats(CSV_FILE, many_thr=1000, low_thr=100)
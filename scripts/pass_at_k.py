import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = "."
OUTPUT_FILE = "pass_at_k_robust.png"

def calculate_pass_at_k(n, c, k):
    """
    Estimates Pass@k.
    n: total samples available
    c: correct samples found
    k: hypothetical budget (Pass@k)
    """
    k_adj = min(n, k)
    
    if c == n: return 1.0
    if c == 0: return 0.0
    
    # prod( (n-c-i) / (n-i) )
    prob_fail = 1.0
    for i in range(k_adj):
        prob_fail *= (n - c - i) / (n - i)
    return 1.0 - prob_fail

def analyze_pass_k_robust():
    print("Loading data for Pass@k...")
    files = glob.glob(f"{DATA_DIR}/*.parquet")
    if not files:
        print("No parquet files found.")
        return

    df = pd.concat([pd.read_parquet(f).assign(dataset=os.path.basename(f).replace('.parquet','')) for f in files], ignore_index=True)
    
    # Group by Model + Example to see the max N
    rollout_counts = df.groupby(['model_id', 'dataset', 'example_id']).size()
    max_k_found = int(rollout_counts.max())
    mean_k_found = rollout_counts.mean()
    
    print(f"DEBUG: Found max {max_k_found} rollouts per example (Mean: {mean_k_found:.1f}).")
    
    target_k = 8 if max_k_found >= 8 else max_k_found
    print(f"DEBUG: Calculating Pass@{target_k}...")

    rollout_stats = df.groupby(['model_id', 'dataset', 'example_id'])['reward'].agg(['count', 'sum']).reset_index()
    rollout_stats.columns = ['model_id', 'dataset', 'example_id', 'n_samples', 'n_correct']
    
    # Calculate scores
    rollout_stats['pass_1'] = rollout_stats['n_correct'] / rollout_stats['n_samples']
    rollout_stats[f'pass_{target_k}'] = rollout_stats.apply(
        lambda x: calculate_pass_at_k(x['n_samples'], x['n_correct'], target_k), axis=1
    )
    
    model_scores = rollout_stats.groupby('model_id')[[ 'pass_1', f'pass_{target_k}']].mean().reset_index()
    model_scores = model_scores.sort_values('pass_1', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    sns.barplot(data=model_scores, x='model_id', y=f'pass_{target_k}', color='#d62728', alpha=0.6, label=f'Pass@{target_k} (Potential)')
    
    sns.barplot(data=model_scores, x='model_id', y='pass_1', color='#1f77b4', alpha=0.9, label='Pass@1 (Baseline)')
    
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Model Potential: Pass@1 vs Pass@{target_k}", fontsize=14, weight='bold')
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_pass_k_robust()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = "../inference-scratch"
OUTPUT_FILE = "pass_at_k_ordered_by_pass1.png"

def calculate_pass_at_k(n, c, k):
    """
    Estimates Pass@k using the unbiased estimator.
    """
    k_adj = min(n, k)
    if c == n: return 1.0
    if c == 0: return 0.0
    
    prob_fail = 1.0
    for i in range(k_adj):
        prob_fail *= (n - c - i) / (n - i)
    
    return 1.0 - prob_fail

def analyze_pass_k_sorted_by_baseline():
    print("Loading data for Pass@k Analysis...")
    
    files = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True)
    
    if not files:
        print(f"No parquet files found in {DATA_DIR}.")
        return

    print(f"Found {len(files)} files. Compiling data...")
    
    df_list = []
    for f in files:
        if os.path.getsize(f) == 0:
            continue
            
        try:
            temp_df = pd.read_parquet(f)
            
            dataset_name = os.path.basename(f).replace('.parquet', '')
            model_id = os.path.basename(os.path.dirname(f))
            
            temp_df['dataset'] = dataset_name
            temp_df['model_id'] = model_id
            
            if 'reward' in temp_df.columns and 'example_id' in temp_df.columns:
                df_list.append(temp_df[['model_id', 'dataset', 'example_id', 'reward']])
                
        except Exception:
            pass

    if not df_list:
        print("Could not load any valid dataframes.")
        return

    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} total rows.")

    rollout_counts = df.groupby(['model_id', 'dataset', 'example_id']).size()
    max_k_found = int(rollout_counts.max())
    target_k = max_k_found 
    
    print(f"DEBUG: Max samples found: {max_k_found}. Calculating Pass@{target_k}...")

    df['is_correct'] = df['reward'] > 0
    rollout_stats = df.groupby(['model_id', 'dataset', 'example_id'])['is_correct'].agg(['count', 'sum']).reset_index()
    rollout_stats.columns = ['model_id', 'dataset', 'example_id', 'n_samples', 'n_correct']
    
    rollout_stats['pass_1'] = rollout_stats['n_correct'] / rollout_stats['n_samples']
    
    rollout_stats[f'pass_{target_k}'] = rollout_stats.apply(
        lambda x: calculate_pass_at_k(x['n_samples'], x['n_correct'], target_k), axis=1
    )
    
    model_scores = rollout_stats.groupby('model_id')[[ 'pass_1', f'pass_{target_k}']].mean().reset_index()
    
    model_scores = model_scores.sort_values('pass_1', ascending=False)

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    sns.barplot(
        data=model_scores, 
        x='model_id', 
        y=f'pass_{target_k}', 
        color='#d62728', 
        alpha=0.6, 
        label=f'Pass@{target_k} (Potential)'
    )
    
    sns.barplot(
        data=model_scores, 
        x='model_id', 
        y='pass_1', 
        color='#1f77b4', 
        alpha=0.9, 
        label='Pass@1 (Baseline)'
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Model Performance: Pass@1 (Sorted) vs Pass@{target_k} Potential", fontsize=16, weight='bold')
    plt.ylabel("Score (Probability of Correct Answer)")
    plt.xlabel("Model")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(OUTPUT_FILE)
    print(f"Saved analysis to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_pass_k_sorted_by_baseline()
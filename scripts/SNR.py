import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = "../inference-scratch"
OUTPUT_FILE = "signal_to_noise_audit.png"

def analyze_signal_to_noise():
    print("Loading data for Signal-to-Noise Audit...")
    
    files = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True)
    
    if not files:
        print(f"No parquet files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} files. Compiling data...")
    
    dfs = []
    for f in files:
        if os.path.getsize(f) == 0:
            continue

        try:
            dataset_name = os.path.basename(f).replace('.parquet', '')
            
            df = pd.read_parquet(f)
            
            df['dataset'] = dataset_name
            
            if 'model_id' not in df.columns:
                folder_name = os.path.basename(os.path.dirname(f))
                df['model_id'] = folder_name
            else:
                # Ensure it's a string for grouping consistency
                df['model_id'] = df['model_id'].astype(str)

            dfs.append(df)
            
        except Exception as e:
            print(f"Skipped {f}: {e}")
            
    if not dfs:
        print("No valid data loaded.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Data loaded: {len(full_df)} total rows across {full_df['dataset'].nunique()} datasets.")


    print("Calculating stability metrics...")
    
    rollout_stats = full_df.groupby(['dataset', 'model_id', 'example_id'])['reward'].std().reset_index()
    
    dataset_noise = rollout_stats.groupby('dataset')['reward'].mean().reset_index()
    dataset_noise.columns = ['dataset', 'noise_score']

    dataset_skill = full_df.groupby('dataset')['reward'].mean().reset_index()
    dataset_skill.columns = ['dataset', 'mean_accuracy']

    audit_df = pd.merge(dataset_noise, dataset_skill, on='dataset')
    
    audit_df['noise_score'] = audit_df['noise_score'].fillna(0)

    
    plt.figure(figsize=(14, 9))
    sns.set_theme(style="whitegrid")

    sns.scatterplot(
        data=audit_df, 
        x='mean_accuracy', 
        y='noise_score', 
        s=150, 
        color='#d62728', 
        edgecolor='black', 
        alpha=0.8
    )

    for i, row in audit_df.iterrows():
        plt.text(
            row['mean_accuracy'] + 0.01, 
            row['noise_score'], 
            row['dataset'], 
            fontsize=9, 
            weight='bold',
            alpha=0.8
        )

    plt.axhline(y=0.25, color='grey', linestyle='--', alpha=0.5, label='High Noise Threshold')
    plt.axvline(x=0.5, color='grey', linestyle=':', alpha=0.5, label='Random Chance')

    plt.text(0.1, 0.45, "THE LOTTERY ZONE\n(High Noise + Low Skill)\nExclude these!", 
             color='red', fontsize=12, fontweight='bold', ha='left', va='top', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.text(0.9, 0.05, "THE GOLD STANDARD\n(Low Noise + High Skill)", 
             color='green', fontsize=12, fontweight='bold', ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title("Dataset Audit: Signal (Skill) vs. Noise (Luck)", fontsize=16, weight='bold')
    plt.xlabel("Mean Accuracy (Higher is Better)", fontsize=12)
    plt.ylabel("Instability / Std Dev (Lower is Better)", fontsize=12)
    
    plt.xlim(0, 1.1)
    plt.ylim(0, max(0.55, audit_df['noise_score'].max() + 0.1))
    
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE)
    print(f"Generated analysis chart: {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_signal_to_noise()
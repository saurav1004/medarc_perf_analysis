import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = "." 
OUTPUT_FILE = "signal_to_noise_audit.png"

def analyze_signal_to_noise():
    print("Loading data...")
    files = glob.glob(f"{DATA_DIR}/*.parquet")
    
    dfs = []
    for f in files:
        name = os.path.basename(f).replace('.parquet', '')
        try:
            df = pd.read_parquet(f)
            df['dataset'] = name
            dfs.append(df)
        except Exception as e:
            print(f"Skipped {name}: {e}")
            
    if not dfs:
        print("No data found.")
        return

    full_df = pd.concat(dfs, ignore_index=True)

    rollout_stats = full_df.groupby(['dataset', 'model_id', 'example_id'])['reward'].std().reset_index()
    
    dataset_noise = rollout_stats.groupby('dataset')['reward'].mean().reset_index()
    dataset_noise.columns = ['dataset', 'noise_score']

    dataset_skill = full_df.groupby('dataset')['reward'].mean().reset_index()
    dataset_skill.columns = ['dataset', 'mean_accuracy']

    audit_df = pd.merge(dataset_noise, dataset_skill, on='dataset')

    plt.figure(figsize=(12, 8))
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

    # Annotate Zones
    plt.text(0.1, 0.45, "THE LOTTERY ZONE\n(High Noise + Low Skill)\nExclude these!", 
             color='red', fontsize=12, fontweight='bold', ha='left', va='top')
    
    plt.text(0.9, 0.05, "THE GOLD STANDARD\n(Low Noise + High Skill)", 
             color='green', fontsize=12, fontweight='bold', ha='right', va='bottom')

    plt.title("Dataset Audit: Signal (Skill) vs. Noise (Luck)", fontsize=16, weight='bold')
    plt.xlabel("Mean Accuracy (Higher is Better)", fontsize=12)
    plt.ylabel("Instability / Std Dev (Lower is Better)", fontsize=12)
    plt.xlim(0, 1.05)
    plt.ylim(0, 0.55)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE)
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_signal_to_noise()
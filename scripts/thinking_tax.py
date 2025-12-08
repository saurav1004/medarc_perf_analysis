import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

DATA_DIR = "."  # Path to your parquet files
OUTPUT_FILE = "thinking_efficiency_frontier.png"

def analyze_thinking_efficiency():
    #    We use MedQA and MedXpertQA as they are standard benchmarks
    target_files = [f"{DATA_DIR}/medqa.parquet", f"{DATA_DIR}/medxpertqa-reasoning.parquet"]
    
    valid_files = [f for f in target_files if os.path.exists(f)]
    if not valid_files:
        print(f"No data found in {DATA_DIR}. Please check the path.")
        return

    dfs = [pd.read_parquet(f) for f in valid_files]
    df = pd.concat(dfs, ignore_index=True)

    #    We create a readable 'Model Family' column
    df['Model Family'] = df['model_id'].apply(lambda x: 
        'Thinking' if 'think' in x.lower() or 'reason' in x.lower() else 'Instruct')
    
    df['Token Bucket'] = (df['model_token_completion'] // 200) * 200
    
    efficiency = df.groupby(['Model Family', 'Token Bucket'])['reward'].mean().reset_index()
    
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=efficiency, 
        x='Token Bucket', 
        y='reward', 
        hue='Model Family', 
        style='Model Family',
        s=100,
        alpha=0.7,
        palette={'Thinking': '#d62728', 'Instruct': '#1f77b4'}
    )
    
    # This uses numpy and does NOT require statsmodels
    if len(efficiency[efficiency['Model Family']=='Thinking']) > 3:
        sns.regplot(
            data=efficiency[efficiency['Model Family']=='Thinking'], 
            x='Token Bucket', y='reward', scatter=False, order=2, 
            color='#d62728', label='_nolegend_', ci=None
        )

    plt.title("The 'Thinking Tax': Efficiency Frontier", fontsize=16, weight='bold')
    plt.xlabel("Inference Cost (Tokens Generated)", fontsize=12)
    plt.ylabel("Accuracy (Mean Reward)", fontsize=12)
    plt.xlim(0, 4000) # Cap at 4k tokens to keep focus on relevant range
    plt.legend(title="Model Type", loc="lower right")
    plt.tight_layout()
    
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_thinking_efficiency()
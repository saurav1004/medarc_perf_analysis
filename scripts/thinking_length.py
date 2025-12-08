import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = "."
OUTPUT_FILE = "thinking_length_correlation.png"

def analyze_thinking_length():
    print("Loading data for Thinking Analysis...")
    files = glob.glob(f"{DATA_DIR}/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    df['is_thinker'] = df['model_id'].apply(lambda x: any(k in x.lower() for k in ['think', 'reason', 'qwq']))
    df_think = df[df['is_thinker']].copy()
    
    if df_think.empty:
        print("No thinking models found.")
        return

    df_think['Outcome'] = df_think['reward'].apply(lambda x: 'Correct' if x > 0 else 'Incorrect')

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    sns.boxplot(
        data=df_think, 
        x='model_id', 
        y='model_token_completion', 
        hue='Outcome', 
        palette={'Correct': '#2ca02c', 'Incorrect': '#d62728'},
        showfliers=False 
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.title("Do Models 'Overthink' when they Fail?", fontsize=14, weight='bold')
    plt.ylabel("Tokens Generated (Thinking Trace)")
    plt.xlabel("Model")
    plt.legend(title="Outcome")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_thinking_length() 
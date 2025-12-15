import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = "../inference-scratch"
OUTPUT_FILE = "thinking_length_correlation.png"

def analyze_thinking_length():
    print("Loading data for Thinking Analysis...")
    
    files = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True)
    
    if not files:
        print(f"No parquet files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} files. Compiling dataframe...")
    
    df_list = []
    for f in files:
        try:
            temp_df = pd.read_parquet(f)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Skipping bad file {f}: {e}")

    if not df_list:
        print("Could not load any dataframes.")
        return

    df = pd.concat(df_list, ignore_index=True)

    df['model_id'] = df['model_id'].astype(str)

    keywords = ['think', 'reason', 'qwq']
    df['is_thinker'] = df['model_id'].apply(lambda x: any(k in x.lower() for k in keywords))
    
    df_think = df[df['is_thinker']].copy()
    
    if df_think.empty:
        print("No thinking models found based on keywords:", keywords)
        print("Available models:", df['model_id'].unique()[:10]) # Debug print
        return

    print(f"Analyzing {df_think['model_id'].nunique()} thinking models...")

    df_think['Outcome'] = df_think['reward'].apply(lambda x: 'Correct' if x > 0 else 'Incorrect')

    plt.figure(figsize=(14, 8)) # Increased size slightly for more models
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
    plt.title("Do Models 'Overthink' when they Fail?", fontsize=16, weight='bold')
    plt.ylabel("Tokens Generated (Thinking Trace)")
    plt.xlabel("Model")
    plt.legend(title="Outcome")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Saved analysis to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_thinking_length()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# MedQA is best for this as it usually has 4-5 options and is large
TARGET_FILE = "./medqa.parquet" 
OUTPUT_FILE = "positional_bias_check.png"

def analyze_bias():
    if not os.path.exists(TARGET_FILE):
        print(f"{TARGET_FILE} not found.")
        return

    df = pd.read_parquet(TARGET_FILE)
    
    top_models = df.groupby('model_id')['reward'].mean().nlargest(6).index
    df_top = df[df['model_id'].isin(top_models)].copy()
    
    valid_answers = ['A', 'B', 'C', 'D', 'E']
    df_top = df_top[df_top['answer'].isin(valid_answers)]

    sns.set_theme(style="ticks", context="paper")
    g = sns.FacetGrid(df_top, col="model_id", col_wrap=3, height=4, sharey=False)
    g.map(sns.countplot, "answer", order=sorted(df_top['answer'].unique()), palette="viridis")
    
    
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Positional Bias Check: Answer Distribution by Model", fontsize=16, weight='bold')
    
    g.savefig(OUTPUT_FILE, dpi=300)
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_bias()
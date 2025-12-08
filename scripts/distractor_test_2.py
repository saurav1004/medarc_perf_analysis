import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

FILE_OP4 = "./medbullets-op4.parquet"
FILE_OP5 = "./medbullets-op5.parquet"
OUTPUT_FILE = "distractor_stress_test.png"

def analyze_distractors():
    if not (os.path.exists(FILE_OP4) and os.path.exists(FILE_OP5)):
        print("MedBullets files missing.")
        return

    df4 = pd.read_parquet(FILE_OP4).groupby('model_id')['reward'].mean().rename("Op4 Accuracy")
    df5 = pd.read_parquet(FILE_OP5).groupby('model_id')['reward'].mean().rename("Op5 Accuracy")
    
    comparison = pd.concat([df4, df5], axis=1).dropna()
    comparison['Performance Drop'] = comparison['Op4 Accuracy'] - comparison['Op5 Accuracy']
    
    comparison = comparison.sort_values('Performance Drop', ascending=False)
    top_movers = comparison.head(5)
    bottom_movers = comparison.tail(5)
    plot_data = pd.concat([top_movers, bottom_movers])

    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(12, 8))
    
    plt.hlines(y=plot_data.index, xmin=plot_data['Op5 Accuracy'], xmax=plot_data['Op4 Accuracy'], color='grey', alpha=0.4)
    plt.scatter(plot_data['Op5 Accuracy'], plot_data.index, color='skyblue', alpha=1, s=150, label='5 Options (Harder)')
    plt.scatter(plot_data['Op4 Accuracy'], plot_data.index, color='navy', alpha=1, s=150, label='4 Options (Easier)')
    
    plt.title("Distractor Stress Test: Sensitivity to Extra Wrong Options", weight='bold')
    plt.xlabel("Accuracy")
    plt.legend()
    plt.grid(axis='x', linestyle='--')
    
    for i, row in plot_data.iterrows():
        diff = row['Performance Drop']
        plt.text((row['Op4 Accuracy'] + row['Op5 Accuracy'])/2, i, f"-{diff:.1%}", 
                 verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_distractors()
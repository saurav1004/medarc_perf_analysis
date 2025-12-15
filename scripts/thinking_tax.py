import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

DATA_DIR = "../inference-scratch"
OUTPUT_FILE = "thinking_efficiency_frontier_final.png"

TARGET_TASKS = [
    'medqa', 
    'medxpertqa-reasoning', 
    'medcalc_bench',    
    'mmlu_pro_health',  
    'm_arc'             
]

FONT_SIZE = 9

try:
    from adjustText import adjust_text
except ImportError:
    print("CRITICAL: 'adjustText' is not installed.")
    print("Run: pip install adjustText")
    exit()

def analyze_efficiency_frontier_final():
    print("Loading data...")
    files = []
    for task in TARGET_TASKS:
        found = glob.glob(f"{DATA_DIR}/**/*{task}*.parquet", recursive=True)
        files.extend(found)
    
    if not files:
        print("No files found.")
        return

    dfs = []
    for f in files:
        if os.path.getsize(f) == 0: continue
        try:
            temp_df = pd.read_parquet(f)
            temp_df['model_id'] = os.path.basename(os.path.dirname(f))
            
            if 'model_token_completion' in temp_df.columns: pass
            elif 'generation_token_count' in temp_df.columns:
                 temp_df['model_token_completion'] = temp_df['generation_token_count']
            else: continue 
            
            dfs.append(temp_df[['model_id', 'reward', 'model_token_completion']])
        except Exception: pass

    if not dfs: return

    df = pd.concat(dfs, ignore_index=True)
    df = df[df['model_token_completion'] <= 8000] # Cap extreme outliers

    model_metrics = df.groupby('model_id').agg(
        Accuracy=('reward', 'mean'),
        Cost=('model_token_completion', 'mean')
    ).reset_index()

    think_keywords = ['think', 'reason', 'qwq', 'intellect']
    model_metrics['Family'] = model_metrics['model_id'].apply(lambda x: 
        'Thinking' if any(k in x.lower() for k in think_keywords) else 'Standard')

    print(f"Plotting {len(model_metrics)} models...")

    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(16, 12))
    
    sns.scatterplot(
        data=model_metrics, x='Cost', y='Accuracy', hue='Family', style='Family',
        s=120, alpha=0.8, palette={'Thinking': '#d62728', 'Standard': '#1f77b4'}, zorder=3
    )

    for family, color in [('Thinking', '#d62728'), ('Standard', '#1f77b4')]:
        subset = model_metrics[model_metrics['Family'] == family]
        if len(subset) >= 4:
            sns.regplot(
                data=subset, x='Cost', y='Accuracy', scatter=False, logx=True, color=color, 
                label=f'{family} Trend', truncate=False, 
                line_kws={'linestyle': '--', 'alpha': 0.5, 'linewidth': 1.5}, ax=plt.gca()
            )

    texts = []
    for _, row in model_metrics.iterrows():
        short_name = row['model_id'].replace('-instruct', '').replace('-reasoning', '-R').replace('gpt-oss-', 'GPT-')
        
        label_color = '#8b0000' if row['Family'] == 'Thinking' else '#1f3f77'
        weight = 'bold' if row['Family'] == 'Thinking' else 'normal'
        
        t = plt.text(
            row['Cost'], 
            row['Accuracy'], 
            short_name, 
            fontsize=FONT_SIZE, 
            weight=weight, 
            color=label_color,
        )
        texts.append(t)

    print("Running adjustText optimization (this may take 10-20 seconds)...")
    adjust_text(
        texts,
        force_points=0.3, 
        force_text=0.5,
        expand_points=(1.5, 1.5), 
        arrowprops=dict(arrowstyle='-', color='grey', alpha=0.5, lw=0.5)
    )

    plt.title("Efficiency Frontier: Full Model Landscape", fontsize=18, weight='bold')
    plt.xlabel("Average Cost (Tokens per Query)", fontsize=14)
    plt.ylabel("Average Accuracy", fontsize=14)
    plt.xlim(0, 4000)
    plt.legend(title="Model Family", loc="lower right", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_efficiency_frontier_final()
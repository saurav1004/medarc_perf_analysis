import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import glob
import os
import json
import numpy as np
from adjustText import adjust_text

DATA_DIR = "inference-scratch"
METADATA_FILE = "model_metadata.json"
OUTPUT_FILE = "token_efficiency.png"

TARGET_TASKS = [
    'medqa', 
    'medxpertqa-reasoning', 
    'medcalc_bench',    
    'mmlu_pro_health',  
    'm_arc'             
]

FONT_SIZE = 9

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, 'r') as f:
        data = json.load(f)
    return data

def analyze_token_efficiency():
    print("Loading data...")
    metadata = load_metadata()
    if not metadata: return

    files = []
    for task in TARGET_TASKS:
        found = glob.glob(f"{DATA_DIR}/**/*{task}*.parquet", recursive=True)
        files.extend(found)
    
    if not files: return

    dfs = []
    for f in files:
        if os.path.getsize(f) == 0: continue
        try:
            temp_df = pd.read_parquet(f)
            mid = os.path.basename(os.path.dirname(f))
            if mid not in metadata: continue

            temp_df['model_id'] = mid
            if 'model_token_completion' in temp_df.columns: pass
            elif 'generation_token_count' in temp_df.columns:
                 temp_df['model_token_completion'] = temp_df['generation_token_count']
            else: continue 
            
            dfs.append(temp_df[['model_id', 'reward', 'model_token_completion']])
        except Exception: pass

    if not dfs: return

    df = pd.concat(dfs, ignore_index=True)
    df = df[df['model_token_completion'] <= 8000] 

    model_metrics = df.groupby('model_id').agg(
        Accuracy=('reward', 'mean'),
        Cost=('model_token_completion', 'mean')
    ).reset_index()

    size_list, family_list = [], []
    for _, row in model_metrics.iterrows():
        info = metadata[row['model_id']]
        family_list.append('Thinking' if info.get('reasoning', False) else 'Standard')
        size_list.append(info.get('size_without_quant', 'Unknown'))

    model_metrics['Category'] = family_list
    model_metrics['Size'] = size_list

    
    size_order = ['Tiny', 'Small', 'Medium', 'Large', 'API', 'Unknown']
    existing_sizes = [s for s in size_order if s in model_metrics['Size'].unique()]
    
    palette = sns.color_palette("plasma", n_colors=len(existing_sizes))
    size_color_map = dict(zip(existing_sizes, palette))
    
    shape_map = {
        'Tiny': 'o', 
        'Small': 'v', 
        'Medium': 's', 
        'Large': 'D', 
        'API': '*', 
    }
    size_markers = {s: shape_map.get(s, 'o') for s in existing_sizes}

    print(f"Plotting {len(model_metrics)} models...")

    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(16, 12))
    
    sns.scatterplot(
        data=model_metrics, 
        x='Cost', 
        y='Accuracy', 
        hue='Size', 
        hue_order=existing_sizes,
        style='Size',
        markers=size_markers,
        palette=size_color_map,
        s=100,
        alpha=0.9,
        edgecolor='black', 
        linewidth=1, 
        zorder=3, 
        legend=False 
    )

    size_handles = []
    for s in existing_sizes:
        h = mlines.Line2D([], [], color=size_color_map[s], marker=size_markers[s], 
                          linestyle='None', markersize=12, label=s,
                          markeredgecolor='black', markeredgewidth=1)
        size_handles.append(h)

    legend1 = plt.legend(handles=size_handles, title="Model Size", loc='lower right', 
                         bbox_to_anchor=(0.98, 0.02), framealpha=0.9, edgecolor='gray', labelspacing=1.2)
    plt.gca().add_artist(legend1)

    cat_handles = [
        mpatches.Patch(color='green', label='Thinking (Green Text)'),
        mpatches.Patch(color='#1f3f77', label='Standard (Blue Text)')
    ]
    plt.legend(handles=cat_handles, title="Label Color", loc='lower right', 
               bbox_to_anchor=(0.98, 0.18), framealpha=0.9, edgecolor='gray')

    DO_NOT_ADJUST = []
    texts_fixed = []
    texts_to_adjust = []

    for _, row in model_metrics.iterrows():
        mid = row['model_id']
        info = metadata.get(mid, {})
        
        label_text = info.get('name', mid)

        label_color = 'green' if row['Category'] == 'Thinking' else '#1f3f77'
        weight = 'bold' if row['Category'] == 'Thinking' else 'bold'

        t = plt.text(
            row['Cost'], 
            row['Accuracy'], 
            label_text, 
            fontsize=FONT_SIZE, 
            weight=weight, 
            color=label_color
        )

        if mid in DO_NOT_ADJUST:
            texts_fixed.append(t)
        else:
            texts_to_adjust.append(t)

    try:
        adjust_text(
            texts_to_adjust, 
            add_objects=texts_fixed,
            force_points=0.3, 
            force_text=0.3, 
            expand_points=(1.5, 1.5), 
            arrowprops=dict(arrowstyle='-', color='grey', alpha=0.5, lw=0.5)
        )
    except Exception: pass

    plt.title("Token Efficiency - Reasoning Heavy Tasks", fontsize=18, weight='bold')
    plt.xlabel("Average Cost (Tokens per Query)", fontsize=14)
    plt.ylabel("Average Accuracy", fontsize=14)
    plt.xlim(0, 4200)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_token_efficiency()
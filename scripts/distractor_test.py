import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = "../inference-scratch"
OUTPUT_FILE = "distractor_stress_test.png"

KEYWORD_OP4 = "medbullets-op4"
KEYWORD_OP5 = "medbullets-op5"

def load_task_data(task_keyword):
    """
    Searches for all parquet files matching a keyword and calculates 
    mean reward (accuracy) per model.
    """
    print(f"Searching for '{task_keyword}' data...")
    files = glob.glob(f"{DATA_DIR}/**/*{task_keyword}*.parquet", recursive=True)
    
    if not files:
        print(f"  -> No files found for {task_keyword}")
        return None

    df_list = []
    for f in files:
        if os.path.getsize(f) == 0: continue
        
        try:
            temp_df = pd.read_parquet(f, columns=['reward']) 
            
            model_id = os.path.basename(os.path.dirname(f))
            temp_df['model_id'] = model_id
            
            df_list.append(temp_df)
        except Exception:
            pass
            
    if not df_list:
        return None
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    return full_df.groupby('model_id')['reward'].mean()

def analyze_distractors():
    acc_op4 = load_task_data(KEYWORD_OP4)
    acc_op5 = load_task_data(KEYWORD_OP5)

    if acc_op4 is None or acc_op5 is None:
        print("Missing data for one or both tasks. Cannot compare.")
        return


    print("Merging datasets...")
    comparison = pd.concat([acc_op4, acc_op5], axis=1, keys=['Op4 Accuracy', 'Op5 Accuracy']).dropna()
    
    if comparison.empty:
        print("No models found that have results for BOTH Op4 and Op5.")
        return

    print(f"Comparing {len(comparison)} models...")


    comparison['Performance Drop'] = comparison['Op4 Accuracy'] - comparison['Op5 Accuracy']
    
    comparison = comparison.sort_values('Performance Drop', ascending=False)
    
    if len(comparison) > 15:
        top_movers = comparison.head(10)
        bottom_movers = comparison.tail(5)
        plot_data = pd.concat([top_movers, bottom_movers])
        # Remove duplicates if list is small
        plot_data = plot_data[~plot_data.index.duplicated(keep='first')]
    else:
        plot_data = comparison

    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(14, 10))
    
    plt.hlines(
        y=plot_data.index, 
        xmin=plot_data['Op5 Accuracy'], 
        xmax=plot_data['Op4 Accuracy'], 
        color='grey', 
        alpha=0.5,
        linewidth=2
    )
    
    plt.scatter(
        plot_data['Op5 Accuracy'], 
        plot_data.index, 
        color='skyblue', 
        alpha=1, 
        s=150, 
        label='5 Options (Harder)'
    )
    
    plt.scatter(
        plot_data['Op4 Accuracy'], 
        plot_data.index, 
        color='navy', 
        alpha=1, 
        s=150, 
        label='4 Options (Easier)'
    )
    
    plt.title("Distractor Stress Test: Sensitivity to Extra Wrong Options", weight='bold', fontsize=16)
    plt.xlabel("Accuracy (Mean Reward)", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.legend(loc='lower left')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        drop_val = row['Performance Drop']
        
        mid_x = (row['Op4 Accuracy'] + row['Op5 Accuracy']) / 2
        
        text_color = '#d62728' if drop_val > 0 else '#2ca02c'
        label_text = f"-{drop_val:.1%}" if drop_val > 0 else f"+{abs(drop_val):.1%}"
        
        plt.text(
            mid_x, 
            i, 
            label_text, 
            verticalalignment='bottom', 
            horizontalalignment='center', 
            color=text_color, 
            fontsize=10,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_distractors()
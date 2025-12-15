import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = "../inference-scratch"
OUTPUT_FILE = "rote_vs_reason_quadrant.png"

KNOWLEDGE_TASKS = [
    'medqa', 
    'medbullets-op5', 
    'medbullets-op4', 
    'pubmedqa', 
    'med_mcqa',             
    'mmlu_pro_health',      
    'metamedqa',            
    'medconceptsqa'         
]

REASONING_TASKS = [
    'medxpertqa-reasoning', 
    'medxpertqa-understanding', 
    'm_arc', 
    'longhealth',               
    'medcalc_bench'             
]

PROMINENT_STANDARD_TAGS = [
    'llama-3-70b',
    'gemma-3-27b',
    'gpt-oss-120b',
    'qwen-next-80b',
    'sonnet-4_5',
    'mistral-large',  # Example, if present
]

def analyze_rote_vs_reason():
    print("Loading data for Rote vs. Reason Analysis...")
    
    all_files = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True)
    
    if not all_files:
        print(f"No files found in {DATA_DIR}")
        return

    knowledge_dfs = []
    reasoning_dfs = []
    
    print(f"Found {len(all_files)} files. Sorting into Knowledge vs. Reasoning...")

    for f in all_files:
        if os.path.getsize(f) == 0:
            continue
            
        filename = os.path.basename(f)
        model_id = os.path.basename(os.path.dirname(f))
        
        try:
            is_knowledge = any(k in filename for k in KNOWLEDGE_TASKS)
            is_reasoning = any(r in filename for r in REASONING_TASKS)
            
            if is_knowledge or is_reasoning:
                df = pd.read_parquet(f)
                df['model_id'] = model_id
                
                # Keep only necessary columns
                df = df[['model_id', 'reward']].copy()
                
                if is_knowledge:
                    knowledge_dfs.append(df)
                elif is_reasoning:
                    reasoning_dfs.append(df)
        
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not knowledge_dfs or not reasoning_dfs:
        print("Missing dataset files.")
        return

    print("Aggregating scores...")
    df_k = pd.concat(knowledge_dfs).groupby('model_id')['reward'].mean().rename("Knowledge Score")
    df_r = pd.concat(reasoning_dfs).groupby('model_id')['reward'].mean().rename("Reasoning Score")
    
    df = pd.concat([df_k, df_r], axis=1).dropna()
    print(f"Successfully analyzed {len(df)} models with complete data.")
    
    # Keyword detection for Thinking models
    think_keywords = ['think', 'reason', 'qwq', 'intellect']
    df['Type'] = df.index.map(lambda x: 'Thinking' if any(k in x.lower() for k in think_keywords) else 'Standard')

    print("\n--- Classification Audit ---")
    thinkers = df[df['Type'] == 'Thinking'].index.tolist()
    print(f"Classified {len(thinkers)} models as 'Thinking'.")

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 12))
    
    sns.scatterplot(
        data=df[df['Type']=='Standard'], 
        x="Knowledge Score", 
        y="Reasoning Score", 
        color='grey', 
        s=150, 
        alpha=0.6, 
        label='Standard'
    )
    
    sns.scatterplot(
        data=df[df['Type']=='Thinking'], 
        x="Knowledge Score", 
        y="Reasoning Score", 
        color='#d62728', 
        s=300, 
        alpha=0.9, 
        label='Thinking'
    )
    
    numeric_df = df[['Knowledge Score', 'Reasoning Score']]
    min_val = min(numeric_df.min())
    max_val = max(numeric_df.max())
    lims = [min_val - 0.05, max_val + 0.05]
    
    plt.plot(lims, lims, '--', color='black', alpha=0.3, label='Balanced Performance')
    
    plt.text(lims[1], lims[0], "Rote Learners\n(High Knowledge, Low Reasoning)", 
             ha='right', va='bottom', fontsize=12, color='grey', fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    plt.text(lims[0], lims[1], "Pure Reasoners\n(Low Knowledge, High Reasoning)", 
             ha='left', va='top', fontsize=12, color='grey', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))


    top_thinkers = df[df['Type'] == 'Thinking'].nlargest(5, 'Reasoning Score')
    for model in top_thinkers.index:
        short_name = model.replace('-instruct', '').replace('-reasoning', '-R')
        plt.text(
            df.loc[model, "Knowledge Score"], 
            df.loc[model, "Reasoning Score"] + 0.01, 
            short_name, 
            fontsize=11, fontweight='bold', color='#a80000', ha='center'
        )

    standard_models = df[df['Type'] == 'Standard']
    for model in standard_models.index:
        if any(tag in model.lower() for tag in PROMINENT_STANDARD_TAGS):
            short_name = model.replace('-instruct', '').replace('gpt-oss-', 'GPT-')
            plt.text(
                df.loc[model, "Knowledge Score"], 
                df.loc[model, "Reasoning Score"] - 0.015,  
                short_name, 
                fontsize=10, fontweight='bold', color='#404040', ha='center'
            )

    plt.title("Rote vs. Reason: Model Capabilities Map", fontsize=18, weight='bold', pad=20)
    plt.xlabel("Knowledge Score (MedQA, PubMed, etc.)", fontsize=14)
    plt.ylabel("Reasoning Score (MedXpert, ARC, LongHealth)", fontsize=14)
    plt.xlim(lims)
    plt.ylim(lims)
    
    plt.legend(title="Model Class", loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Generated chart: {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_rote_vs_reason()
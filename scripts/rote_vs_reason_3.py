import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = "./"
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

def analyze_rote_vs_reason():
    all_files = glob.glob(f"{DATA_DIR}/*.parquet")
    
    knowledge_dfs = []
    reasoning_dfs = []
    
    print("Loading data...")
    for f in all_files:
        filename = f.split('/')[-1]
        if any(k in filename for k in KNOWLEDGE_TASKS):
            knowledge_dfs.append(pd.read_parquet(f))
        elif any(r in filename for r in REASONING_TASKS):
            reasoning_dfs.append(pd.read_parquet(f))
            
    if not knowledge_dfs or not reasoning_dfs:
        print("Missing dataset files. Check your DATA_DIR.")
        return

    df_k = pd.concat(knowledge_dfs).groupby('model_id')['reward'].mean().rename("Knowledge Score")
    df_r = pd.concat(reasoning_dfs).groupby('model_id')['reward'].mean().rename("Reasoning Score")
    
    df = pd.concat([df_k, df_r], axis=1).dropna()
    print(f"Analyzed {len(df)} models.")
    
    df['Type'] = df.index.map(lambda x: 'Thinking' if 'think' in x.lower() or 'reason' in x.lower() else 'Standard')

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 10))
    
    ax = sns.scatterplot(
        data=df, x="Knowledge Score", y="Reasoning Score", 
        hue="Type", size="Type", sizes=(150, 300),
        palette={'Thinking': '#d62728', 'Standard': 'grey'},
        alpha=0.8
    )
    
    numeric_df = df[['Knowledge Score', 'Reasoning Score']]
    min_val = numeric_df.min().min()
    max_val = numeric_df.max().max()
    lims = [min_val - 0.05, max_val + 0.05]
    
    plt.plot(lims, lims, '--', color='black', alpha=0.3, label='Balanced Performance')
    
    plt.text(lims[1], lims[0], "Rote Learners\n(High Knowledge, Low IQ)", 
             ha='right', va='bottom', fontsize=12, color='grey', fontweight='bold')
    plt.text(lims[0], lims[1], "Pure Reasoners\n(Low Knowledge, High IQ)", 
             ha='left', va='top', fontsize=12, color='grey', fontweight='bold')

    top_thinkers = df[df['Type'] == 'Thinking'].nlargest(5, 'Reasoning Score')
    for model in top_thinkers.index:
        short_name = model.split('-think')[0].split('-reason')[0]
        plt.text(df.loc[model, "Knowledge Score"], df.loc[model, "Reasoning Score"] + 0.005, 
                 short_name, fontsize=10, fontweight='bold', color='#a80000')

    plt.title("Rote vs. Reason: Model Capabilities Map", fontsize=16, weight='bold', pad=20)
    plt.xlabel("Knowledge Score (MedQA, PubMed)", fontsize=12)
    plt.ylabel("Reasoning Score (MedXpert, ARC)", fontsize=12)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.legend(title="Model Class", loc='lower right')
    plt.tight_layout()
    
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_rote_vs_reason()
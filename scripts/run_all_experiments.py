import subprocess
import os
from pathlib import Path
import re

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

# We expect directories inside data/ to contain a .csv file of the same name (or similar)
DATASETS = [
    ("eclipse", "data/eclipse/final dataset for work ecllipse.csv"),
    ("mozilla_firefox", "data/mozilla_firefox/mozilla_firefox.csv"),
    ("mozilla_core", "data/mozilla_core/mozilla_core.csv"),
    ("thunderbird", "data/thunderbird/thunderbird.csv")
]

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    results = {}
    
    for project_name, csv_path in DATASETS:
        if not os.path.exists(csv_path):
            print(f"[ERROR] CSV not found for {project_name} at {csv_path}")
            continue
            
        model_path = MODELS_DIR / f"ltr_ranker_{project_name}.json"
        
        print("\n" + "="*80)
        print(f"=== RUNNING EXPERIMENTS FOR: {project_name.upper()} ===")
        print("="*80)
        
        # 1. Train the model
        print(f"\n---> Training LTR model for {project_name}...")
        train_cmd = [
            "python", "scripts/train_ltr.py",
            "--csv-path", csv_path,
            "--model-output", str(model_path),
            "--skip-cv-gate"  # Force saving even if it doesn't beat baseline in CV
        ]
        
        # We don't capture output here so the user can see progress bars in Colab
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Training failed for {project_name}")
            continue
            
        # 2. Evaluate the model
        print(f"\n---> Evaluating models for {project_name}...")
        eval_cmd = [
            "python", "scripts/compare_models.py",
            "--csv-path", csv_path,
            "--model-path", str(model_path)
        ]
        
        try:
            # We capture output to parse the final metrics for the table
            result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            # Extract metrics using regex
            # Example format:
            #   Top-1 Accuracy        10.00%    23.00%  +13.00%
            #   MRR                   0.2198    0.3277  +0.1078
            wsm_top1 = re.search(r'Top-1 Accuracy\s+([\d.]+)%\s+([\d.]+)%', result.stdout)
            wsm_top3 = re.search(r'Top-3 Accuracy\s+([\d.]+)%\s+([\d.]+)%', result.stdout)
            wsm_top5 = re.search(r'Top-5 Accuracy\s+([\d.]+)%\s+([\d.]+)%', result.stdout)
            mrr = re.search(r'MRR\s+([\d.]+)\s+([\d.]+)', result.stdout)
            
            if wsm_top1 and wsm_top3 and wsm_top5 and mrr:
                results[project_name] = {
                    "wsm_top1": wsm_top1.group(1),
                    "ltr_top1": wsm_top1.group(2),
                    "wsm_top3": wsm_top3.group(1),
                    "ltr_top3": wsm_top3.group(2),
                    "wsm_top5": wsm_top5.group(1),
                    "ltr_top5": wsm_top5.group(2),
                    "wsm_mrr": mrr.group(1),
                    "ltr_mrr": mrr.group(2)
                }
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Evaluation failed for {project_name}")
            print(e.stderr)
            continue
            
    # 3. Print LaTeX Table
    print("\n\n" + "="*80)
    print("=== FINAL LATEX TABLE RESULTS ===")
    print("="*80 + "\n")
    
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Performance Comparison of WSM Baseline and XGBRanker (LTR) Across Open-Source Repositories}")
    print(r"\label{tab:multi_project_results}")
    print(r"\renewcommand{\arraystretch}{1.2}")
    print(r"\begin{tabular}{lcccccccc}")
    print(r"\toprule")
    print(r" & \multicolumn{2}{c}{\textbf{Top-1 Acc.\ (\%)}} & \multicolumn{2}{c}{\textbf{Top-3 Acc.\ (\%)}} & \multicolumn{2}{c}{\textbf{Top-5 Acc.\ (\%)}} & \multicolumn{2}{c}{\textbf{MRR}} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    print(r"\textbf{Project} & \textbf{WSM} & \textbf{LTR} & \textbf{WSM} & \textbf{LTR} & \textbf{WSM} & \textbf{LTR} & \textbf{WSM} & \textbf{LTR} \\")
    print(r"\midrule")
    
    display_names = {
        "eclipse": "Eclipse IDE",
        "mozilla_firefox": "Mozilla Firefox",
        "mozilla_core": "Mozilla Core",
        "thunderbird": "Thunderbird"
    }
    
    for proj in ["eclipse", "mozilla_firefox", "mozilla_core", "thunderbird"]:
        if proj in results:
            r = results[proj]
            print(f"{display_names[proj]} & {r['wsm_top1']} & \\textbf{{{r['ltr_top1']}}} & {r['wsm_top3']} & \\textbf{{{r['ltr_top3']}}} & {r['wsm_top5']} & \\textbf{{{r['ltr_top5']}}} & {r['wsm_mrr']} & \\textbf{{{r['ltr_mrr']}}} \\\\")
            
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")

if __name__ == "__main__":
    main()

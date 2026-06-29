"""
scripts/run_multi_eval.py — Automate training and evaluation across multiple projects.

This script runs the LTR pipeline (train + eval) for:
1. Eclipse IDE
2. Mozilla Firefox
3. Mozilla Core
4. Thunderbird

And outputs a Markdown table for the research paper.
"""

import subprocess
import sys
from pathlib import Path

# Paths to datasets and where to save their respective models
PROJECTS = [
    {
        "name": "Eclipse IDE",
        "csv": "data/eclipse/final dataset for work ecllipse.csv",
        "model": "models/ltr_ranker_eclipse.json",
        "test_size": 200,
    },
    {
        "name": "Mozilla Firefox",
        "csv": "data/mozilla_firefox/mozilla_firefox.csv",
        "model": "models/ltr_ranker_firefox.json",
        "test_size": 1000, # Using larger test set for larger datasets
    },
    {
        "name": "Mozilla Core",
        "csv": "data/mozilla_core/mozilla_core.csv",
        "model": "models/ltr_ranker_core.json",
        "test_size": 1000,
    },
    {
        "name": "Thunderbird",
        "csv": "data/thunderbird/thunderbird.csv",
        "model": "models/ltr_ranker_thunderbird.json",
        "test_size": 500,
    },
]

def run_command(cmd: list[str], capture: bool = True) -> str:
    print(f"Running: {' '.join(cmd)}", flush=True)
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"Error running command:\n{result.stderr if capture else 'Check logs above'}")
        sys.exit(1)
    return result.stdout if capture else ""

def parse_metrics(output: str) -> dict:
    """Parse the stdout of compare_models.py to extract metrics."""
    metrics = {}
    lines = output.splitlines()
    for line in lines:
        if "Top-1 Accuracy" in line:
            parts = line.split()
            metrics["wsm_top1"] = parts[2]
            metrics["ltr_top1"] = parts[3]
        elif "Top-3 Accuracy" in line:
            parts = line.split()
            metrics["wsm_top3"] = parts[2]
            metrics["ltr_top3"] = parts[3]
        elif "Top-5 Accuracy" in line:
            parts = line.split()
            metrics["wsm_top5"] = parts[2]
            metrics["ltr_top5"] = parts[3]
        elif "MRR" in line and "Latency" not in line:
            parts = line.split()
            metrics["wsm_mrr"] = parts[1]
            metrics["ltr_mrr"] = parts[2]
    return metrics

def main():
    print("=" * 60)
    print("Multi-Project Evaluation Pipeline")
    print("=" * 60)
    
    results = {}
    
    for proj in PROJECTS:
        name = proj["name"]
        csv_path = proj["csv"]
        model_path = proj["model"]
        test_size = proj["test_size"]
        
        print(f"\nProcessing {name}...")
        
        # 1. Train Model (skip CV gate to ensure automation doesn't block)
        train_cmd = [
            sys.executable, "scripts/train_ltr.py",
            "--csv-path", csv_path,
            "--model-output", model_path,
            "--test-size", str(test_size),
            "--skip-cv-gate"
        ]
        run_command(train_cmd, capture=False)
        
        # 2. Evaluate Model
        eval_cmd = [
            sys.executable, "scripts/compare_models.py",
            "--csv-path", csv_path,
            "--model-path", model_path,
            "--test-size", str(test_size)
        ]
        eval_output = run_command(eval_cmd, capture=True)
        metrics = parse_metrics(eval_output)
        
        results[name] = metrics
        print(f"Completed {name}: Top-1 LTR = {metrics.get('ltr_top1')}")

    # 3. Print the Markdown Table
    print("\n\n" + "=" * 60)
    print("FINAL RESULTS FOR PAPER (Table 5)")
    print("=" * 60)
    print("""
\\begin{table*}[t]
\\centering
\\caption{Performance Comparison of WSM Baseline and XGBRanker (LTR) Across Open-Source Repositories}
\\label{tab:multi_project_results}
\\renewcommand{\\arraystretch}{1.2}
\\begin{tabular}{lcccccccc}
\\toprule
 & \\multicolumn{2}{c}{\\textbf{Top-1 Accuracy (\\%)}} & \\multicolumn{2}{c}{\\textbf{Top-3 Accuracy (\\%)}} & \\multicolumn{2}{c}{\\textbf{Top-5 Accuracy (\\%)}} & \\multicolumn{2}{c}{\\textbf{MRR}} \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}
\\textbf{Project / Repository} & \\textbf{WSM} & \\textbf{LTR} & \\textbf{WSM} & \\textbf{LTR} & \\textbf{WSM} & \\textbf{LTR} & \\textbf{WSM} & \\textbf{LTR} \\\\
\\midrule""")

    for proj in PROJECTS:
        name = proj["name"]
        m = results.get(name, {})
        # Strip the % signs for LaTeX
        w_t1 = m.get('wsm_top1', '0').replace('%', '')
        l_t1 = m.get('ltr_top1', '0').replace('%', '')
        w_t3 = m.get('wsm_top3', '0').replace('%', '')
        l_t3 = m.get('ltr_top3', '0').replace('%', '')
        w_t5 = m.get('wsm_top5', '0').replace('%', '')
        l_t5 = m.get('ltr_top5', '0').replace('%', '')
        w_mrr = m.get('wsm_mrr', '0')
        l_mrr = m.get('ltr_mrr', '0')
        
        print(f"{name} & {w_t1} & \\textbf{{{l_t1}}} & {w_t3} & \\textbf{{{l_t3}}} & {w_t5} & \\textbf{{{l_t5}}} & {w_mrr} & \\textbf{{{l_mrr}}} \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table*}
""")

if __name__ == "__main__":
    main()

from csv_pipeline import load_dataset_from_csv, build_developers_from_stats
from github_pipeline import fetch_github_issues, process_github_issues
from bug_assigner import BugAssigner, Bug

def run_unified_pipeline():
    print("=== Step 1: Loading Offline Eclipse Data ===")
    dataset_filepath = "archive/final dataset for work ecllipse.csv"
    try:
        eclipse_bugs, eclipse_dev_stats = load_dataset_from_csv(dataset_filepath)
        print(f"Loaded {len(eclipse_bugs)} bugs from Eclipse dataset.")
    except Exception as e:
        print(f"Failed to load Eclipse CSV: {e}")
        eclipse_bugs, eclipse_dev_stats = [], {}

    print("\n=== Step 2: Fetching Live VS Code Data ===")
    raw_issues = fetch_github_issues(repo="microsoft/vscode", limit=100)
    vscode_bugs, vscode_dev_stats = process_github_issues(raw_issues)
    print(f"Loaded {len(vscode_bugs)} bugs from VS Code GitHub.")

    print("\n=== Step 3: Merging Datasets ===")
    all_historical_bugs = eclipse_bugs + vscode_bugs
    
    # Merge developer stats. If by crazy chance a developer has the same name in both, 
    # this will overwrite, but for these distinct datasets, it's safe.
    combined_stats = {**eclipse_dev_stats, **vscode_dev_stats}
    
    all_developers, all_resolutions = build_developers_from_stats(combined_stats)
    
    print(f"Total Unified Historical Bugs: {len(all_historical_bugs)}")
    print(f"Total Unique Developers in Pool: {len(all_developers)}")

    print("\n=== Step 4: Training Bug Assigner AI ===")
    assigner = BugAssigner()
    assigner.fit_historical_data(all_historical_bugs, all_developers, all_resolutions)
    print("AI Training Complete.")

    print("\n=== Step 5: Routing a Custom New Bug ===")
    # Let's create a custom bug to test the routing
    new_bug = Bug(
        id=999999, 
        description="The VS Code integrated terminal is completely frozen and crashes when I type git status.", 
        severity="critical", 
        module="terminal"
    )
    
    print(f"[CUSTOM BUG]: {new_bug.description}")
    print(f"Severity: {new_bug.severity} | Module: {new_bug.module}")
    
    best_dev, rankings = assigner.assign_bug(new_bug, k=10)
    
    if best_dev:
        print(f"\n  -> Optimal Assignment: {best_dev.name} (Score: {rankings[0][1]:.3f})")
        print("  -> Top 5 Candidates:")
        for dev, score in rankings[:5]:
            print(f"       - {dev.name}: {score:.3f} (Exp: {dev.experience}, Load: {dev.workload}%, Fix Time: {dev.fix_time:.1f}h)")
    else:
        print("  -> No assignment could be made.")

if __name__ == "__main__":
    run_unified_pipeline()

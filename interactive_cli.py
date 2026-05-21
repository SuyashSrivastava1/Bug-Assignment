import sys
from csv_pipeline import load_dataset_from_csv
from bug_assigner import BugAssigner, Developer, Bug

def build_real_developers(dev_stats):
    """
    Build developers strictly using the real metrics extracted from the CSV,
    without any random 'dummy' data.
    """
    developers = []
    resolutions = {}
    
    dev_id_counter = 1
    for assignee, stats in dev_stats.items():
        # REAL METRICS:
        experience = stats["bugs_fixed"]
        avg_fix_time = stats["total_fix_time"] / stats["bugs_fixed"]
        # Approximate domain skill based on how many distinct components they work on
        domain_skill = min(100, 50 + len(stats["components"]) * 10)
        
        # STATIC NEUTRAL METRICS: 
        # (Since we don't have open issue counts or reopen counts in the offline CSV,
        # we set them all to perfectly neutral baselines rather than using random dummy data)
        success_rate = 100 
        workload = 0 
        
        dev = Developer(
            id=dev_id_counter, 
            name=stats["name"], 
            experience=experience, 
            fix_time=avg_fix_time, 
            success_rate=success_rate, 
            workload=workload, 
            domain_skill=domain_skill
        )
        developers.append(dev)
        
        for bug_id in stats["resolved_bug_ids"]:
            resolutions[bug_id] = dev_id_counter
            
        dev_id_counter += 1
        
    return developers, resolutions

def main():
    print("=========================================")
    print("   Interactive Bug Assignment CLI")
    print("=========================================")
    print("Initializing system with real offline data...")
    
    dataset_filepath = "archive/final dataset for work ecllipse.csv"
    try:
        all_bugs, dev_stats = load_dataset_from_csv(dataset_filepath)
        print(f"Loaded {len(all_bugs)} bugs successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find {dataset_filepath}. Please ensure the data exists.")
        sys.exit(1)
        
    # Build developers using ONLY real metrics
    developers, resolutions = build_real_developers(dev_stats)
    
    print(f"Extracted {len(developers)} real developers.")
    print("Training AI model (this takes a few seconds)...")
    
    assigner = BugAssigner()
    # We train on ALL historical data for maximum accuracy in the CLI
    assigner.fit_historical_data(all_bugs, developers, resolutions)
    
    print("AI Training Complete!\n")
    print("Type your bug description below, and the AI will find the absolute best")
    print("real developer from the Eclipse dataset to fix it.")
    print("Type 'exit' or 'quit' to close the program.")
    
    bug_counter = 999000
    while True:
        print("\n-----------------------------------------")
        user_input = input("Enter Bug Description:\n> ")
        
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue
            
        # Create a bug object from user input
        bug_counter += 1
        new_bug = Bug(
            id=bug_counter,
            description=user_input,
            severity="critical", # Defaulting to critical for testing speed
            module="general"
        )
        
        print("\nAnalyzing and routing...")
        best_dev, rankings = assigner.assign_bug(new_bug, k=5)
        
        if best_dev:
            print(f"\n[SUCCESS] OPTIMAL ASSIGNMENT: **{best_dev.name}**")
            print(f"   Utility Score: {rankings[0][1]:.3f}")
            print(f"   Developer's Real Stats: Fixed {best_dev.experience} bugs | Avg Fix Time: {best_dev.fix_time:.1f} hours")
            
            print("\n   Next Best Candidates:")
            for dev, score in rankings[1:5]:
                print(f"     - {dev.name} (Score: {score:.3f} | Fixed: {dev.experience} bugs)")
        else:
            print("[ERROR] No assignment could be made.")

if __name__ == "__main__":
    main()

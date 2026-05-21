import csv
import datetime
import random
from bug_assigner import BugAssigner, Developer, Bug

def load_dataset_from_csv(filepath):
    print(f"Loading bugs from {filepath}...")
    bugs = []
    dev_stats = {}
    
    # This assumes a typical academic bug dataset CSV structure.
    # You may need to adjust the column names to match your downloaded dataset!
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # Extract basic data
            bug_id = row.get("Bug ID")
            summary = row.get("Summary", "")
            description = "" # The eclipse dataset uses Summary instead of a long description body
            severity = row.get("Severity", "normal")
            component = row.get("Component", "general")
            assignee = row.get("Assignee Real Name") or row.get("Assignee")
            
            # Skip unassigned bugs
            if not assignee or assignee == "nobody@mozilla.org":
                continue
                
            # Parse dates to calculate fix time
            try:
                # Adjust the date format string based on your specific CSV
                created = datetime.datetime.strptime(row.get("Opened", ""), "%Y-%m-%d %H:%M:%S")
                resolved = datetime.datetime.strptime(row.get("Changed", ""), "%Y-%m-%d %H:%M:%S")
                fix_time_hours = max(0.1, (resolved - created).total_seconds() / 3600.0)
            except (ValueError, TypeError):
                fix_time_hours = 24.0 # Fallback if dates are missing or format differs
                
            # Map severity
            severity_lower = severity.lower()
            if severity_lower in ['blocker', 'critical', 'major']:
                mapped_severity = 'critical'
            elif severity_lower in ['trivial', 'minor']:
                mapped_severity = 'minor'
            else:
                mapped_severity = 'normal'
                
            # Create the Bug object
            bug_obj = Bug(bug_id, f"{summary} {description}", mapped_severity, component)
            bugs.append(bug_obj)
            
            # Track Developer stats
            if assignee not in dev_stats:
                dev_stats[assignee] = {
                    "name": assignee,
                    "bugs_fixed": 0,
                    "total_fix_time": 0.0,
                    "components": set(),
                    "resolved_bug_ids": []
                }
                
            dev_stats[assignee]["bugs_fixed"] += 1
            dev_stats[assignee]["total_fix_time"] += fix_time_hours
            dev_stats[assignee]["components"].add(component)
            dev_stats[assignee]["resolved_bug_ids"].append(bug_id)
            
    return bugs, dev_stats

def build_developers_from_stats(dev_stats):
    developers = []
    resolutions = {}
    
    dev_id_counter = 1
    for assignee, stats in dev_stats.items():
        experience = stats["bugs_fixed"]
        avg_fix_time = stats["total_fix_time"] / stats["bugs_fixed"]
        success_rate = random.randint(85, 100) # Synthesized if not in CSV
        workload = random.randint(10, 90) # Synthesized if not in CSV
        domain_skill = min(100, 50 + len(stats["components"]) * 10)
        
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

if __name__ == "__main__":
    # INSTRUCTIONS:
    # 1. Download a bug dataset (e.g., Eclipse or Mozilla Bugzilla dataset) from Kaggle
    # 2. Place the .csv file in this directory and update the filepath below
    dataset_filepath = "archive/final dataset for work ecllipse.csv" 
    
    try:
        all_bugs, dev_stats = load_dataset_from_csv(dataset_filepath)
        print(f"Successfully loaded {len(all_bugs)} assigned bugs.")
        
        developers, resolutions = build_developers_from_stats(dev_stats)
        
        # Split data: 80% history to train the AI, 20% to test it
        split_index = int(len(all_bugs) * 0.8)
        historical_bugs = all_bugs[:split_index]
        test_bugs = all_bugs[split_index:]
        
        print(f"Training BugAssigner with {len(historical_bugs)} historical bugs...")
        assigner = BugAssigner()
        assigner.fit_historical_data(historical_bugs, developers, resolutions)
        
        print(f"Evaluating {len(test_bugs)} new bugs...")
        
        # Evaluate a single random test bug
        test_bug = random.choice(test_bugs)
        print(f"\n[NEW BUG TO ROUTE]: {test_bug.description[:100]}...")
        best_dev, rankings = assigner.assign_bug(test_bug, k=5)
        
        if best_dev:
            print(f"Optimal Assignment: {best_dev.name} (Score: {rankings[0][1]:.3f})")
        else:
            print("No assignment could be made.")
            
    except FileNotFoundError:
        print(f"Error: Could not find {dataset_filepath}.")
        print("Please download a dataset and place it in the project folder first!")

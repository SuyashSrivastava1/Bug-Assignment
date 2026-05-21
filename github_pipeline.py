import urllib.request
import json
import datetime
import random
import numpy as np

# Import the logic we built previously
from bug_assigner import BugAssigner, Developer, Bug

def fetch_github_issues(repo="microsoft/vscode", limit=100):
    print(f"Fetching resolved issues for {repo} from GitHub API...")
    url = f"https://api.github.com/repos/{repo}/issues?state=closed&per_page={limit}"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []
        
    return data

def process_github_issues(raw_issues):
    bugs = []
    dev_stats = {}
    
    for i in raw_issues:
        # Determine the developer who fixed the issue/PR
        if "pull_request" in i:
            # For PRs, the author of the PR is the developer who wrote the code
            dev_username = i["user"]["login"]
        else:
            # For standard issues, we need someone formally assigned
            assignees = i.get("assignees", [])
            if not assignees:
                continue
            dev_username = assignees[0]["login"]
            
        bug_id = i["number"]
        summary = i["title"]
        description = i.get("body") or summary
        
        # Determine severity/component based on labels
        labels = [l["name"].lower() for l in i.get("labels", [])]
        
        if any(l in ["bug", "critical", "crash"] for l in labels):
            severity = "critical"
        elif any(l in ["feature-request", "enhancement"] for l in labels):
            severity = "minor"
        else:
            severity = "normal"
            
        component = "general"
        for l in labels:
            if l.startswith("area-") or l.startswith("component-"):
                component = l
                break
                
        # Calculate fix time in hours
        try:
            created = datetime.datetime.strptime(i["created_at"], "%Y-%m-%dT%H:%M:%SZ")
            resolved = datetime.datetime.strptime(i["closed_at"], "%Y-%m-%dT%H:%M:%SZ")
            fix_time_hours = max(0.1, (resolved - created).total_seconds() / 3600.0)
        except (ValueError, TypeError):
            fix_time_hours = 24.0 # Fallback
            
        
        bug_obj = Bug(bug_id, summary + " " + description, severity, component)
        bugs.append(bug_obj)
        
        # Aggregate Developer Stats
        if dev_username not in dev_stats:
            dev_stats[dev_username] = {
                "name": dev_username,
                "bugs_fixed": 0,
                "total_fix_time": 0.0,
                "components": set()
            }
            
        dev_stats[dev_username]["bugs_fixed"] += 1
        dev_stats[dev_username]["total_fix_time"] += fix_time_hours
        dev_stats[dev_username]["components"].add(component)
        
        # Store who fixed what to pass to the assigner
        dev_stats[dev_username].setdefault("resolved_bug_ids", []).append(bug_id)
        
    return bugs, dev_stats

def build_developers_and_history(dev_stats):
    developers = []
    resolutions = {}
    
    dev_id_counter = 1
    for username, stats in dev_stats.items():
        # Synthesize metrics based on real data
        experience = stats["bugs_fixed"] # Number of bugs acts as experience proxy
        avg_fix_time = stats["total_fix_time"] / stats["bugs_fixed"]
        
        # We don't have regression data from a single API call, so we synthesize a plausible success rate
        success_rate = random.randint(85, 100)
        
        # Workload (0-100%) - Synthesized for the simulation
        workload = random.randint(10, 90)
        
        # Domain Skill: Let's assume high skill (90+) if they fixed many bugs in many components
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
    # 1. Fetch real GitHub issues
    # Using VS Code repo as it's a massive, highly active open source project
    raw_issues = fetch_github_issues(repo="microsoft/vscode", limit=100)
    
    if not raw_issues:
        print("Failed to fetch bugs.")
        exit(1)
        
    # 2. Process data into internal format
    all_bugs, dev_stats = process_github_issues(raw_issues)
    
    if len(all_bugs) < 5:
        print("Not enough bugs with assignees found in the recent 100 closed issues.")
        exit(1)
        
    print(f"Processed {len(all_bugs)} bugs assigned to {len(dev_stats)} unique developers.")
    
    # 3. Build developers and history map
    developers, resolutions = build_developers_and_history(dev_stats)
    
    # Let's use the first 80% as historical data, and test the algorithm on the remaining 20%
    split_index = int(len(all_bugs) * 0.8)
    historical_bugs = all_bugs[:split_index]
    test_bugs = all_bugs[split_index:]
    
    print(f"\n--- Initializing Bug Assigner with {len(historical_bugs)} historical bugs ---")
    assigner = BugAssigner()
    assigner.fit_historical_data(historical_bugs, developers, resolutions)
    
    print(f"\n--- Testing AI Assigner on new bugs ---")
    
    # Randomly select a few test bugs to show
    sample_test_bugs = random.sample(test_bugs, min(3, len(test_bugs)))
    
    for bug in sample_test_bugs:
        print(f"\n[NEW BUG]: {bug.description.splitlines()[0] if bug.description else bug.description}")
        print(f"Severity: {bug.severity} | Module: {bug.module}")
        
        best_dev, rankings = assigner.assign_bug(bug, k=5)
        
        if best_dev:
            print(f"  -> Optimal Assignment: {best_dev.name} (Score: {rankings[0][1]:.3f})")
            print("  -> Top 3 Candidates:")
            for dev, score in rankings[:3]:
                print(f"       - {dev.name}: {score:.3f} (Exp: {dev.experience}, Load: {dev.workload}%, Fix Time: {dev.fix_time:.1f}h)")
            
            # Show who actually fixed it in reality for comparison!
            actual_dev_id = resolutions.get(bug.id)
            if actual_dev_id:
                actual_dev = next((d for d in developers if d.id == actual_dev_id), None)
                if actual_dev:
                    print(f"  -> *Real-world Assigner was*: {actual_dev.name}")
        else:
            print("  -> No assignment could be made.")

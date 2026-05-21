import sys
import re
import urllib.request
import json
import datetime

from bug_assigner import BugAssigner, Developer, Bug
from github_pipeline import fetch_github_issues, process_github_issues

def build_real_developers(dev_stats):
    """Build developers without dummy/random data."""
    developers = []
    resolutions = {}
    
    dev_id_counter = 1
    for username, stats in dev_stats.items():
        experience = stats["bugs_fixed"]
        avg_fix_time = stats["total_fix_time"] / stats["bugs_fixed"]
        domain_skill = min(100, 50 + len(stats["components"]) * 10)
        
        # Real neutral baselines instead of random data
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
        
        for bug_id in stats.get("resolved_bug_ids", []):
            resolutions[bug_id] = dev_id_counter
            
        dev_id_counter += 1
        
    return developers, resolutions

def parse_github_url(url):
    """Extract owner/repo and issue_number from a github URL."""
    # Matches: https://github.com/microsoft/vscode/issues/1234
    match = re.search(r"github\.com/([^/]+/[^/]+)/issues/(\d+)", url)
    if match:
        return match.group(1), match.group(2)
    return None, None

def fetch_single_issue(repo, issue_number):
    print(f"Fetching specific issue #{issue_number} from {repo}...")
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        print(f"Error fetching specific issue: {e}")
        return None

def main():
    print("=========================================")
    print("   Dynamic GitHub Bug Router")
    print("=========================================")
    
    url = input("Enter a GitHub Issue URL (e.g. https://github.com/microsoft/vscode/issues/212260):\n> ")
    repo, issue_number = parse_github_url(url.strip())
    
    if not repo or not issue_number:
        print("Invalid GitHub URL format. Please ensure it looks like: https://github.com/owner/repo/issues/123")
        sys.exit(1)
        
    # 1. Fetch the target bug
    issue_data = fetch_single_issue(repo, issue_number)
    if not issue_data:
        sys.exit(1)
        
    summary = issue_data.get("title", "")
    description = issue_data.get("body") or summary
    
    labels = [l["name"].lower() for l in issue_data.get("labels", [])]
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
            
    target_bug = Bug(id=int(issue_number), description=f"{summary} {description}", severity=severity, module=component)
    print(f"\n[TARGET BUG]: {summary}")
    print(f"Severity: {severity} | Module: {component}")
    
    # 2. Fetch history for the exact same repository dynamically
    print("\n-----------------------------------------")
    print(f"Automatically collecting historical developer data from {repo}...")
    raw_history = fetch_github_issues(repo, limit=100) # Fetching 100 recent closed issues
    
    if not raw_history:
        print("Failed to fetch historical data from repository.")
        sys.exit(1)
        
    historical_bugs, dev_stats = process_github_issues(raw_history)
    print(f"Extracted {len(historical_bugs)} resolved historical bugs and found {len(dev_stats)} unique developers.")
    
    # 3. Build developers without dummy data
    developers, resolutions = build_real_developers(dev_stats)
    
    # 4. Load massive offline dataset strictly for historical context (not for developers)
    print("\n-----------------------------------------")
    print("Loading massive offline dataset (Eclipse) for deeper AI context...")
    from csv_pipeline import load_dataset_from_csv
    try:
        eclipse_bugs, _ = load_dataset_from_csv("archive/final dataset for work ecllipse.csv")
        print(f"Loaded {len(eclipse_bugs)} historical bugs for AI context matching.")
    except Exception as e:
        print(f"Failed to load offline data: {e}")
        eclipse_bugs = []
        
    combined_history = historical_bugs + eclipse_bugs
    
    # 5. Train AI
    print(f"\nTraining AI model on {len(combined_history)} historical bugs...")
    print(f"Strictly limiting assignment candidates to the {len(developers)} developers from {repo}...")
    assigner = BugAssigner()
    
    # Notice we pass the combined massive history, but ONLY the GitHub developers and GitHub resolutions!
    assigner.fit_historical_data(combined_history, developers, resolutions)
    
    # 6. Route the bug
    print("\n-----------------------------------------")
    print("Analyzing target bug against massive historical data...")
    best_dev, rankings = assigner.assign_bug(target_bug, k=5)
    
    if best_dev:
        print(f"\n[SUCCESS] OPTIMAL ASSIGNMENT: **{best_dev.name}**")
        print(f"   Utility Score: {rankings[0][1]:.3f}")
        print(f"   Developer's Repo Stats: Fixed {best_dev.experience} recent bugs | Avg Fix Time: {best_dev.fix_time:.1f} hours")
        
        print("\n   Next Best Candidates from this Repo:")
        for dev, score in rankings[1:5]:
            print(f"     - {dev.name} (Score: {score:.3f} | Fixed: {dev.experience} bugs)")
    else:
        print("\n[ERROR] No assignment could be made. (Likely not enough historical data related to this issue)")

if __name__ == "__main__":
    main()

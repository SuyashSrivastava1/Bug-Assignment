import urllib.request
import json
import datetime
import random
import numpy as np

# Import the logic we built previously
from bug_assigner import BugAssigner, Developer, Bug

def fetch_bugzilla_data(limit=100, product="Firefox"):
    print(f"Fetching {limit} resolved bugs for {product} from Mozilla Bugzilla...")
    url = f"https://bugzilla.mozilla.org/rest/bug?product={product}&bug_status=RESOLVED&resolution=FIXED&limit={limit}"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        
    return data.get("bugs", [])

def process_bugs(raw_bugs):
    bugs = []
    dev_stats = {}
    
    for b in raw_bugs:
        bug_id = b["id"]
        summary = b["summary"]
        severity = b["severity"]
        if severity == "--":
            severity = "normal"
        component = b["component"]
        
        # Calculate fix time in hours
        try:
            created = datetime.datetime.strptime(b["creation_time"], "%Y-%m-%dT%H:%M:%SZ")
            resolved = datetime.datetime.strptime(b["last_change_time"], "%Y-%m-%dT%H:%M:%SZ")
            fix_time_hours = max(0.1, (resolved - created).total_seconds() / 3600.0)
        except ValueError:
            fix_time_hours = 24.0 # Fallback
            
        assignee = b.get("assigned_to_detail", {})
        dev_email = assignee.get("email", "unknown")
        dev_name = assignee.get("real_name", dev_email)
        
        if dev_email == "nobody@mozilla.org" or not dev_email:
            continue
            
        # Create Bug object
        # Map Bugzilla severities to our internal severity format (critical, minor, normal)
        severity_lower = severity.lower()
        if severity_lower in ['blocker', 'critical', 'major']:
            mapped_severity = 'critical'
        elif severity_lower in ['trivial', 'minor']:
            mapped_severity = 'minor'
        else:
            mapped_severity = 'normal'
            
        bug_obj = Bug(bug_id, summary, mapped_severity, component)
        bugs.append(bug_obj)
        
        # Aggregate Developer Stats
        if dev_email not in dev_stats:
            dev_stats[dev_email] = {
                "name": dev_name,
                "bugs_fixed": 0,
                "total_fix_time": 0.0,
                "components": set()
            }
            
        dev_stats[dev_email]["bugs_fixed"] += 1
        dev_stats[dev_email]["total_fix_time"] += fix_time_hours
        dev_stats[dev_email]["components"].add(component)
        
        # Store who fixed what to pass to the assigner
        dev_stats[dev_email].setdefault("resolved_bug_ids", []).append(bug_id)
        
    return bugs, dev_stats

def build_developers_and_history(dev_stats):
    developers = []
    resolutions = {}
    
    dev_id_counter = 1
    for email, stats in dev_stats.items():
        # Synthesize metrics based on real data
        experience = stats["bugs_fixed"] # Number of bugs acts as experience proxy
        avg_fix_time = stats["total_fix_time"] / stats["bugs_fixed"]
        
        # We don't have regression data from a single API call, so we synthesize a plausible success rate
        success_rate = random.randint(85, 100)
        
        # Workload (0-100%) - Synthesized for the simulation
        workload = random.randint(10, 90)
        
        # Domain Skill: Let's assume high skill (90+) if they fixed many bugs in many components, 
        # or we just assign a baseline and rely on the assignment logic. For this simulation, 
        # we'll say skill = 50 + min(50, len(components) * 10)
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
    # 1. Fetch real Bugzilla data
    raw_bugs = fetch_bugzilla_data(limit=150, product="Firefox")
    
    if not raw_bugs:
        print("Failed to fetch bugs.")
        exit(1)
        
    # 2. Process data into internal format
    all_bugs, dev_stats = process_bugs(raw_bugs)
    
    print(f"Processed {len(all_bugs)} bugs assigned to {len(dev_stats)} unique developers.")
    
    # 3. Build developers and history map
    developers, resolutions = build_developers_and_history(dev_stats)
    
    # Let's use the first 90% as historical data, and test the algorithm on the remaining 10%
    split_index = int(len(all_bugs) * 0.9)
    historical_bugs = all_bugs[:split_index]
    test_bugs = all_bugs[split_index:]
    
    print(f"\n--- Initializing Bug Assigner with {len(historical_bugs)} historical bugs ---")
    assigner = BugAssigner()
    assigner.fit_historical_data(historical_bugs, developers, resolutions)
    
    print(f"\n--- Testing AI Assigner on {len(test_bugs)} new bugs ---")
    
    # Randomly select a few test bugs to show
    import random
    sample_test_bugs = random.sample(test_bugs, min(3, len(test_bugs)))
    
    for bug in sample_test_bugs:
        print(f"\n[NEW BUG]: {bug.description}")
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

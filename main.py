"""
main.py — Single entry point for the Bug Assignment system.

Two modes (chosen interactively at startup):

  1. GitHub Mode
     Paste a GitHub issue URL.  The system fetches the bug details and the
     repository's contributor history automatically, then returns the most
     suitable developer from that project.

     Example URL: https://github.com/facebook/react/issues/36469

  2. Custom Mode
     Type a plain-English bug description.  The system uses the offline
     Eclipse dataset as training context and returns the best-matching
     Eclipse developer.

Usage
-----
    python main.py
"""

import sys
from src.router import Router, parse_github_issue_url, build_bug_from_github_issue
from src.models.bug import Bug


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

DIVIDER = "-" * 50


def print_header():
    print()
    print("=" * 50)
    print("   Bug-to-Developer Assignment System")
    print("=" * 50)


def print_result(best_dev, rankings, weights=None):
    if not best_dev:
        print("\n[ERROR] No assignment could be made.")
        print("  Possible reasons:")
        print("  - Not enough historical data related to this bug.")
        print("  - The repository may not use assignees or PRs.")
        return

    if weights is not None:
        print(f"\n  [Dynamic AI Weights Used] :")
        print(f"  Exp: {weights[0]:.2f} | FixTime: {weights[1]:.2f} | Success: {weights[2]:.2f} | Workload: {weights[3]:.2f} | Domain: {weights[4]:.2f}")

    print(f"\n  BEST MATCH : {best_dev.name}")
    print(f"  Score      : {rankings[0][1]:.3f}")
    print(f"  Bugs Fixed : {best_dev.experience}")
    print(f"  Avg Fix    : {best_dev.fix_time:.1f} hours")

    if len(rankings) > 1:
        print(f"\n  Other Candidates:")
        for dev, score in rankings[1:5]:
            print(f"    - {dev.name:<30} score={score:.3f}  bugs_fixed={dev.experience}")


# ---------------------------------------------------------------------------
# Mode 1: GitHub URL
# ---------------------------------------------------------------------------

def run_github_mode():
    print(f"\n{DIVIDER}")
    url = input("Paste a GitHub issue URL:\n> ").strip()

    repo, issue_number = parse_github_issue_url(url)
    if not repo or not issue_number:
        print("\n[ERROR] Invalid URL. Expected format:")
        print("  https://github.com/owner/repo/issues/123")
        return

    print(f"\n[1/3] Fetching issue #{issue_number} from {repo}...")
    bug = build_bug_from_github_issue(repo, issue_number)
    if not bug:
        print("[ERROR] Could not fetch the issue. Check the URL and try again.")
        return

    print(f"\n  Title    : {bug.description[:120]}")
    print(f"  Severity : {bug.severity}")
    print(f"  Module   : {bug.module}")

    print(f"\n[2/3] Building developer pool from {repo} history...")
    router = Router()
    try:
        router.train_from_github(repo, limit=100)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        return

    print(f"\n[3/3] Finding best developer...")
    print(DIVIDER)
    best_dev, rankings, weights = router.assign(bug, k=5)
    print_result(best_dev, rankings, weights)


# ---------------------------------------------------------------------------
# Mode 2: Custom bug description
# ---------------------------------------------------------------------------

def run_custom_mode():
    print(f"\n{DIVIDER}")
    print("The system will use the offline Eclipse dataset for matching.")
    print("Type your bug description (press Enter twice to submit):")
    print("> ", end="", flush=True)

    lines = []
    while True:
        line = input()
        if line == "" and lines:
            break
        lines.append(line)

    description = " ".join(lines).strip()
    if not description:
        print("[ERROR] No description entered.")
        return

    severity_input = input("\nSeverity? [critical / normal / minor] (default: normal): ").strip().lower()
    severity = severity_input if severity_input in ("critical", "normal", "minor") else "normal"

    bug = Bug(id=0, description=description, severity=severity, module="general")

    print("\n[1/2] Training on offline Eclipse dataset...")
    router = Router()
    router.train_from_csv()

    print("\n[2/2] Finding best developer...")
    print(DIVIDER)
    best_dev, rankings, weights = router.assign(bug, k=5)
    print_result(best_dev, rankings, weights)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print_header()

    while True:
        print(f"\n{DIVIDER}")
        print("Choose a mode:")
        print("  1 — Route a GitHub issue URL")
        print("  2 — Route a custom bug description")
        print("  q — Quit")
        choice = input("> ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("Goodbye.")
            break
        elif choice == "1":
            run_github_mode()
        elif choice == "2":
            run_custom_mode()
        else:
            print("Please enter 1, 2, or q.")


if __name__ == "__main__":
    main()

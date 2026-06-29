"""
main.py — Entry point for the Bug-to-Developer Assignment System.

Two modes (interactive or via flags):

  1. GitHub Mode  — Fetches a GitHub issue and the repo's contributor history,
                    then returns the best developer from that project.

     python main.py --github https://github.com/facebook/react/issues/36469

  2. Custom Mode  — Accepts a plain-English bug description. Uses the offline
                    Eclipse + Bugzilla datasets for NLP context and returns the
                    closest matching Eclipse developer.

     python main.py --custom

Running without flags launches the interactive menu.
"""

from __future__ import annotations

import argparse
import sys
from src.router import Router, parse_github_issue_url, build_bug_from_github_issue
from src.models.bug import Bug

import os

# Initialize Windows virtual terminal processing for ANSI coloring
if os.name == "nt":
    os.system("")

# ANSI formatting codes
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

DIVIDER = f"{CYAN}" + "─" * 60 + f"{RESET}"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bug-to-Developer Assignment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --github https://github.com/facebook/react/issues/36469\n"
            "  python main.py --custom\n"
            "  python main.py --github URL --no-bugzilla   # skip Bugzilla for faster startup\n"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--github",
        metavar="URL",
        help="GitHub issue URL to fetch and assign.",
    )
    group.add_argument(
        "--custom",
        action="store_true",
        help="Enter a plain-text bug description interactively.",
    )
    parser.add_argument(
        "--no-bugzilla",
        action="store_true",
        help=(
            "Skip loading the Bugzilla corpus (203k bugs). "
            "Makes first-run startup ~20x faster at a small accuracy cost. "
            "Recommended until the context cache has been built."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_header() -> None:
    print()
    print(f"{CYAN}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}║{RESET}   {BOLD}{WHITE}Bug-to-Developer Assignment System{RESET}                     {CYAN}║{RESET}")
    print(f"{CYAN}║{RESET}   {BLUE}Hybrid Semantic Prototype + XGBRanker Pipeline{RESET}         {CYAN}║{RESET}")
    print(f"{CYAN}╚══════════════════════════════════════════════════════════╝{RESET}")


def print_result(best_dev, rankings, weights=None, signals=None) -> None:
    if not best_dev:
        print(f"\n  {RED}{BOLD}[ERROR]{RESET} {RED}No assignment could be made.{RESET}")
        print("  Reasons: insufficient historical data, or no contributors found.")
        return

    if signals:
        print(f"\n  {CYAN}┌─ Semantic Signals ───────────────────────────────────────{RESET}")
        labels = ["Experience", "Fix Time", "Success Rate", "Workload", "Domain Skill", "KNN Affinity"]
        for label, w in zip(labels, weights):
            bar = f"{CYAN}█{RESET}" * int(w * 28)
            print(f"  {CYAN}│{RESET}  {WHITE}{label:<14}{RESET} {YELLOW}{w:.3f}{RESET}  {bar}")
        print(f"  {CYAN}└──────────────────────────────────────────────────────────{RESET}")

    print(f"\n  {GREEN}┌─ Assignment Result ──────────────────────────────────────{RESET}")
    print(f"  {GREEN}│{RESET}  {BOLD}{WHITE}BEST MATCH{RESET}  : {GREEN}{BOLD}{best_dev.name}{RESET}")
    print(f"  {GREEN}│{RESET}  Score       : {YELLOW}{rankings[0][1]:.4f}{RESET}")
    print(f"  {GREEN}│{RESET}  Bugs Fixed  : {WHITE}{best_dev.experience}{RESET}")
    print(f"  {GREEN}│{RESET}  Avg Fix Time: {WHITE}{best_dev.fix_time:.1f} hours{RESET}")

    if len(rankings) > 1:
        print(f"  {GREEN}├─ Other Candidates ───────────────────────────────────────{RESET}")
        for dev, score in rankings[1:5]:
            print(f"  {GREEN}│{RESET}  {dev.name:<30} score={YELLOW}{score:.4f}{RESET}")
    print(f"  {GREEN}└──────────────────────────────────────────────────────────{RESET}\n")


# ---------------------------------------------------------------------------
# Mode 1: GitHub URL
# ---------------------------------------------------------------------------

def run_github_mode(url: str, use_bugzilla: bool = True) -> None:
    print(f"\n{DIVIDER}")
    repo, issue_number = parse_github_issue_url(url)
    if not repo or not issue_number:
        print(f"\n  {RED}{BOLD}[ERROR]{RESET} {RED}Invalid URL. Expected format:{RESET}")
        print("    https://github.com/owner/repo/issues/123\n")
        return

    print(f"\n  {MAGENTA}[1/3]{RESET} Fetching issue #{issue_number} from {repo}...")
    bug = build_bug_from_github_issue(repo, issue_number)
    if not bug:
        print(f"  {RED}{BOLD}[ERROR]{RESET} {RED}Could not fetch the issue. Check the URL and try again.{RESET}")
        return

    print(f"  {CYAN}Title{RESET}    : {WHITE}{bug.description[:100]}...{RESET}")
    print(f"  {CYAN}Severity{RESET} : {YELLOW}{bug.severity}{RESET}")
    print(f"  {CYAN}Module{RESET}   : {YELLOW}{bug.module}{RESET}")

    print(f"\n  {MAGENTA}[2/3]{RESET} Building developer pool from {repo} history...")
    router = Router(use_bugzilla=use_bugzilla)
    try:
        router.train_from_github(repo, limit=100)
    except ValueError as e:
        print(f"\n  {RED}{BOLD}[ERROR]{RESET} {RED}{e}{RESET}")
        return

    print(f"\n  {MAGENTA}[3/3]{RESET} Finding best developer...")
    print(DIVIDER)
    best_dev, rankings, weights, signals = router.assign(bug, k=5)
    print_result(best_dev, rankings, weights, signals)


# ---------------------------------------------------------------------------
# Mode 2: Custom bug description
# ---------------------------------------------------------------------------

def run_custom_mode(use_bugzilla: bool = True) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {CYAN}Using offline Eclipse + Bugzilla datasets for NLP context.{RESET}")
    print("  Type your bug description (press Enter twice to submit):\n  > ", end="", flush=True)

    lines: list[str] = []
    while True:
        line = input()
        if line == "" and lines:
            break
        lines.append(line)

    description = " ".join(lines).strip()
    if not description:
        print(f"  {RED}{BOLD}[ERROR]{RESET} {RED}No description entered.{RESET}")
        return

    raw = input(f"\n  Severity? [{YELLOW}critical{RESET} / {YELLOW}normal{RESET} / {YELLOW}minor{RESET}] (default: normal): ").strip().lower()
    severity = raw if raw in ("critical", "normal", "minor") else "normal"

    bug = Bug(id=0, description=description, severity=severity, module="general")

    print(f"\n  {MAGENTA}[1/2]{RESET} Loading offline Eclipse + Bugzilla context...")
    router = Router(use_bugzilla=use_bugzilla)
    router.train_from_csv()

    print(f"\n  {MAGENTA}[2/2]{RESET} Finding best developer...")
    print(DIVIDER)
    best_dev, rankings, weights, signals = router.assign(bug, k=5)
    print_result(best_dev, rankings, weights, signals)


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def run_interactive(use_bugzilla: bool = True) -> None:
    print_header()
    while True:
        print(f"\n{DIVIDER}")
        print(f"  {BOLD}{WHITE}Choose a mode:{RESET}")
        print(f"  {CYAN}1{RESET} — Route a GitHub issue URL")
        print(f"  {CYAN}2{RESET} — Route a custom bug description")
        print(f"  {RED}q{RESET} — Quit")
        choice = input("  > ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print(f"\n  {BOLD}{YELLOW}Goodbye.{RESET}\n")
            break
        elif choice == "1":
            url = input(f"\n  Paste a GitHub issue URL:\n  > ").strip()
            run_github_mode(url, use_bugzilla=use_bugzilla)
        elif choice == "2":
            run_custom_mode(use_bugzilla=use_bugzilla)
        else:
            print(f"  {RED}Please enter 1, 2, or q.{RESET}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    use_bugzilla = not args.no_bugzilla

    if args.github:
        print_header()
        run_github_mode(args.github, use_bugzilla=use_bugzilla)
    elif args.custom:
        print_header()
        run_custom_mode(use_bugzilla=use_bugzilla)
    else:
        run_interactive(use_bugzilla=use_bugzilla)


if __name__ == "__main__":
    main()

"""
scripts/build_cache.py
======================
Pre-builds the SentenceTransformer embeddings cache.

Two modes:
  python scripts/build_cache.py                 # Eclipse only (~2 min) → cache_context_10000.pkl
  python scripts/build_cache.py --with-bugzilla # Eclipse + Bugzilla (~2 hrs) → cache_context_213425.pkl

After the cache is built once, every `python main.py` run loads it in seconds.

Usage with Bugzilla (full accuracy):
    python scripts/build_cache.py --with-bugzilla

Usage without Bugzilla (faster, still good):
    python scripts/build_cache.py
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.loaders import csv_loader, bugzilla_loader
from src.loaders.base import build_developers
from src.models.assigner import BugAssigner

ECLIPSE_CSV   = Path("data") / "eclipse" / "final dataset for work ecllipse.csv"
BUGZILLA_TXT  = Path("data") / "bugzilla" / "corpus (fixsev).txt"

parser = argparse.ArgumentParser()
parser.add_argument("--with-bugzilla", action="store_true",
                    help="Include the Bugzilla corpus (203k bugs) for full accuracy. Takes ~2 hours.")
args = parser.parse_args()

print("=" * 60)
print("  Bug Classification — Cache Builder")
print(f"  Mode: {'Eclipse + Bugzilla (full)' if args.with_bugzilla else 'Eclipse only (fast)'}")
print("=" * 60)

# --- Load Eclipse ---
print(f"\n[1/{'4' if args.with_bugzilla else '3'}] Loading Eclipse dataset...")
bugs, dev_stats = csv_loader.load(ECLIPSE_CSV)
developers, resolutions = build_developers(dev_stats)
print(f"      Loaded {len(bugs):,} bugs, {len(developers):,} developers.")

context_bugs = list(bugs)

# --- Load Bugzilla (optional) ---
if args.with_bugzilla:
    print(f"\n[2/4] Loading Bugzilla corpus from {BUGZILLA_TXT} ...")
    if not BUGZILLA_TXT.exists():
        print(f"      [ERROR] File not found: {BUGZILLA_TXT}")
        print("      Download it from: kaggle.com/datasets/qicongliu/bugzilla-bug-reports")
        sys.exit(1)
    bugzilla_bugs, _ = bugzilla_loader.load(BUGZILLA_TXT)
    print(f"      Loaded {len(bugzilla_bugs):,} Bugzilla bug descriptions.")
    # Bugzilla goes FIRST so it is the stable context prefix
    context_bugs = bugzilla_bugs + list(bugs)
    n_context = len(context_bugs)
    step = 3
else:
    n_context = len(context_bugs)
    step = 2

total = len(context_bugs)
print(f"\n[{step}/{step+1}] Fitting assigner — encoding {total:,} sentences ...")
if args.with_bugzilla:
    print("      This will take approximately 2 hours on CPU. Leave it running.")
else:
    print("      This will take approximately 2 minutes on CPU.")
print()

assigner = BugAssigner()
assigner.fit(
    context_bugs,
    developers,
    resolutions,
    n_context=n_context,
)

print(f"\n[{step+1}/{step+1}] Done! Cache saved successfully.")
if args.with_bugzilla:
    print("      You can now run WITHOUT --no-bugzilla for full accuracy:")
    print("        python main.py --github <URL>")
else:
    print("      You can now run:")
    print("        python main.py --github <URL> --no-bugzilla")
print()

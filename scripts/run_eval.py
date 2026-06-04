"""
scripts/run_eval.py — Multi-project evaluation script.

Trains both WSM and LTR on any dataset (local CSV or GitHub repo) and
prints a side-by-side performance comparison ready to copy into the paper.

Run from the project root:
    # Local CSV
    python scripts/run_eval.py --csv "data/eclipse/final dataset for work ecllipse.csv" --test-size 200

    # GitHub repository
    python scripts/run_eval.py --github "facebook/react" --limit 500 --test-size 50
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.loaders import csv_loader, github_loader
from src.loaders.base import build_developers
from src.models.assigner import BugAssigner
from src.models.ltr_data_builder import LTRDataBuilder
from src.models.ltr_trainer import LTRTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate bug assignment models on any repository.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv",    type=str, help="Path to a local CSV dataset.")
    group.add_argument("--github", type=str, help="GitHub repo in 'owner/repo' format.")
    parser.add_argument("--limit",     type=int, default=1000, help="Max GitHub issues to fetch.")
    parser.add_argument("--test-size", type=int, default=200,  help="Number of bugs for the test set.")
    parser.add_argument("--seed",      type=int, default=42,   help="Random seed for reproducibility.")
    return parser.parse_args()


def evaluate(
    assigner: BugAssigner,
    test_bugs: list,
    all_resolutions: dict,
    use_ltr: bool = False,
) -> tuple[float, float, float, float, float, int]:
    assigner._use_ltr = use_ltr
    top1 = top3 = top5 = mrr_sum = no_assignment = 0

    t_start = time.time()
    for bug in test_bugs:
        actual_dev_id = all_resolutions.get(bug.id)
        if actual_dev_id is None:
            no_assignment += 1
            continue

        best, rankings, _, _ = assigner.assign(bug, k=20)
        if best is None:
            no_assignment += 1
            continue

        ranked_ids = [dev.id for dev, _ in rankings]
        actual_pos = next(
            (r for r, did in enumerate(ranked_ids, 1) if did == actual_dev_id),
            None,
        )
        if actual_pos is not None:
            mrr_sum += 1.0 / actual_pos
            if actual_pos == 1: top1 += 1
            if actual_pos <= 3: top3 += 1
            if actual_pos <= 5: top5 += 1

    t_eval    = time.time() - t_start
    evaluated = len(test_bugs) - no_assignment

    return (
        (top1 / evaluated * 100) if evaluated else 0.0,
        (top3 / evaluated * 100) if evaluated else 0.0,
        (top5 / evaluated * 100) if evaluated else 0.0,
        (mrr_sum / evaluated)    if evaluated else 0.0,
        (t_eval / max(evaluated, 1)) * 1000,
        evaluated,
    )


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("   Multi-Project Bug Assignment Evaluation")
    print("=" * 60)

    # 1. Load data
    if args.csv:
        csv_path = Path(args.csv)
        print(f"\n[1/5] Loading CSV: {csv_path.name}...")
        try:
            bugs, dev_stats = csv_loader.load(csv_path)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        project_name = csv_path.stem
    else:
        print(f"\n[1/5] Fetching from GitHub: {args.github}...")
        bugs, dev_stats = github_loader.load(args.github, limit=args.limit)
        project_name = args.github.replace("/", "_")

    devs, resolutions = build_developers(dev_stats)
    resolvable = [b for b in bugs if b.id in resolutions]

    if not resolvable:
        print("[ERROR] No resolvable bugs found in this dataset.")
        sys.exit(1)

    test_size = min(args.test_size, len(resolvable) // 4)
    if test_size == 0:
        print("[ERROR] Dataset too small. Need at least 4 resolvable bugs.")
        sys.exit(1)
    if test_size < args.test_size:
        print(f"[WARNING] Reduced test size to {test_size} (dataset only has {len(resolvable)} bugs).")

    print(f"  Bugs: {len(resolvable):,} resolvable  |  Developers: {len(devs):,}")

    # 2. Split
    rng = random.Random(args.seed)
    rng.shuffle(resolvable)
    test_bugs  = resolvable[:test_size]
    train_bugs = resolvable[test_size:]

    print(f"\n[2/5] Split — Train: {len(train_bugs):,}  |  Test: {test_size:,}")

    # 3. Fit WSM
    print("\n[3/5] Fitting BugAssigner WSM baseline...")
    assigner = BugAssigner()
    t0 = time.time()
    assigner.fit(train_bugs, devs, resolutions)
    print(f"  Fit completed in {time.time()-t0:.1f}s.")

    # 4. Train LTR
    print("\n[4/5] Training project-specific XGBRanker...")
    t0 = time.time()
    builder = LTRDataBuilder(assigner)
    X, y, groups, _ = builder.build_training_data(train_bugs, resolutions)
    if len(X) == 0:
        print("[ERROR] Feature matrix is empty. Cannot train LTR.")
        sys.exit(1)

    trainer = LTRTrainer()
    trainer.train(X, y, groups)
    tmp_model = Path("models") / f"_tmp_{project_name}.json"
    tmp_model.parent.mkdir(exist_ok=True)
    trainer.save_model(str(tmp_model))
    assigner.load_ltr_model(str(tmp_model))
    print(f"  LTR trained in {time.time()-t0:.1f}s.")

    # 5. Evaluate
    print(f"\n[5/5] Evaluating on {test_size} test bugs...")
    wsm = evaluate(assigner, test_bugs, resolutions, use_ltr=False)
    ltr = evaluate(assigner, test_bugs, resolutions, use_ltr=True)

    # Cleanup temp model
    tmp_model.unlink(missing_ok=True)
    (tmp_model.with_suffix(".meta.json")).unlink(missing_ok=True)

    # Print results
    print("\n" + "=" * 60)
    print(f"   RESULTS — {project_name.upper()}")
    print("=" * 60)
    print(f"  {'Metric':<18} {'WSM':>12} {'LTR':>12} {'Gain':>10}")
    print("  " + "-" * 56)
    print(f"  {'Top-1 Accuracy':<18} {wsm[0]:>11.2f}% {ltr[0]:>11.2f}% {ltr[0]-wsm[0]:>+9.2f}%")
    print(f"  {'Top-3 Accuracy':<18} {wsm[1]:>11.2f}% {ltr[1]:>11.2f}% {ltr[1]-wsm[1]:>+9.2f}%")
    print(f"  {'Top-5 Accuracy':<18} {wsm[2]:>11.2f}% {ltr[2]:>11.2f}% {ltr[2]-wsm[2]:>+9.2f}%")
    print(f"  {'MRR':<18} {wsm[3]:>12.4f} {ltr[3]:>12.4f} {ltr[3]-wsm[3]:>+10.4f}")
    print(f"  {'Latency (ms/bug)':<18} {wsm[4]:>10.1f}ms {ltr[4]:>10.1f}ms {ltr[4]-wsm[4]:>+8.1f}ms")
    print("=" * 60)
    print("\nCopy these values into the LaTeX table in documentation/full_paper.tex\n")


if __name__ == "__main__":
    main()

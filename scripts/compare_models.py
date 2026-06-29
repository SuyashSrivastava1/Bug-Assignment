"""
scripts/compare_models.py — Side-by-side evaluation: WSM Baseline vs XGBRanker (LTR).

Run from the project root:
    python scripts/compare_models.py
    python scripts/compare_models.py --test-size 500
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.csv_loader import load as csv_load
from src.loaders.base import build_developers
from src.models.assigner import BugAssigner

DEFAULT_CSV   = Path("data") / "eclipse" / "final dataset for work ecllipse.csv"
DEFAULT_MODEL = Path("models") / "ltr_ranker.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare WSM and LTR model performance side-by-side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv-path",    type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model-path",  type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-size",   type=int,  default=200)
    parser.add_argument("--seed",        type=int,  default=42)
    return parser.parse_args()


def evaluate(
    assigner: BugAssigner,
    test_bugs: list,
    all_resolutions: dict,
    use_ltr: bool = False,
) -> tuple[float, float, float, float, float]:
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
            (rank for rank, dev_id in enumerate(ranked_ids, 1) if dev_id == actual_dev_id),
            None,
        )

        if actual_pos is not None:
            mrr_sum += 1.0 / actual_pos
            if actual_pos == 1: top1 += 1
            if actual_pos <= 3: top3 += 1
            if actual_pos <= 5: top5 += 1

    t_eval    = time.time() - t_start
    evaluated = len(test_bugs) - no_assignment

    top1_pct  = (top1 / evaluated * 100) if evaluated else 0.0
    top3_pct  = (top3 / evaluated * 100) if evaluated else 0.0
    top5_pct  = (top5 / evaluated * 100) if evaluated else 0.0
    mrr       = (mrr_sum / evaluated)    if evaluated else 0.0
    latency   = (t_eval / max(evaluated, 1)) * 1000

    return top1_pct, top3_pct, top5_pct, mrr, latency


def main() -> None:
    args = parse_args()

    print("=" * 55)
    print("  Bug Assignment — WSM vs LTR Model Comparison")
    print("=" * 55)

    # Load & split
    bugs, dev_stats = csv_load(args.csv_path)
    devs, resolutions = build_developers(dev_stats)

    resolvable = [b for b in bugs if b.id in resolutions]
    resolvable.sort(key=lambda b: getattr(b, "created_at", 0.0))

    train_bugs = resolvable[:-args.test_size]
    test_bugs  = resolvable[-args.test_size:]

    print(f"\n  Train: {len(train_bugs):,}  |  Test: {len(test_bugs):,}")

    # Fit assigner
    assigner = BugAssigner()
    assigner.fit(train_bugs, devs, resolutions)

    if not args.model_path.exists():
        print(f"\n  [WARNING] LTR model not found at {args.model_path}.")
        print("  Run  python scripts/train_ltr.py  first to train the model.")
        sys.exit(1)

    assigner.load_ltr_model(str(args.model_path))

    # Evaluate both
    print("\n  Evaluating WSM Baseline...")
    wsm = evaluate(assigner, test_bugs, resolutions, use_ltr=False)
    print("  Evaluating LTR (XGBRanker)...")
    ltr = evaluate(assigner, test_bugs, resolutions, use_ltr=True)

    # Print comparison table
    print("\n" + "=" * 55)
    print(f"  {'Metric':<18} {'WSM':>10} {'LTR':>10} {'Diff':>8}")
    print("  " + "-" * 50)
    print(f"  {'Top-1 Accuracy':<18} {wsm[0]:>9.2f}% {ltr[0]:>9.2f}% {ltr[0]-wsm[0]:>+7.2f}%")
    print(f"  {'Top-3 Accuracy':<18} {wsm[1]:>9.2f}% {ltr[1]:>9.2f}% {ltr[1]-wsm[1]:>+7.2f}%")
    print(f"  {'Top-5 Accuracy':<18} {wsm[2]:>9.2f}% {ltr[2]:>9.2f}% {ltr[2]-wsm[2]:>+7.2f}%")
    print(f"  {'MRR':<18} {wsm[3]:>10.4f} {ltr[3]:>10.4f} {ltr[3]-wsm[3]:>+8.4f}")
    print(f"  {'Latency (ms/bug)':<18} {wsm[4]:>9.1f}ms {ltr[4]:>9.1f}ms {ltr[4]-wsm[4]:>+7.1f}ms")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()

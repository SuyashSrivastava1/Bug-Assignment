"""
scripts/train_ltr.py — Train the XGBRanker Learning-to-Rank model.

Run from the project root:
    python scripts/train_ltr.py
    python scripts/train_ltr.py --csv-path "data/eclipse/final dataset for work ecllipse.csv"
    python scripts/train_ltr.py --test-size 500 --model-output models/ltr_ranker.json
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.csv_loader import load as csv_load
from src.loaders.base import build_developers
from src.models.assigner import BugAssigner
from src.models.ltr_data_builder import LTRDataBuilder
from src.models.ltr_trainer import LTRTrainer

DEFAULT_CSV   = Path("data") / "eclipse" / "final dataset for work ecllipse.csv"
DEFAULT_MODEL = Path("models") / "ltr_ranker.json"
DEFAULT_BASELINE_TOP1 = 22.00  # WSM baseline — LTR CV must beat this to save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the XGBRanker LTR model for bug assignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to the CSV bug dataset.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL,
        help="Where to save the trained XGBRanker model.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Number of bugs held out for testing (not used in training).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split.",
    )
    parser.add_argument(
        "--skip-cv-gate",
        action="store_true",
        help="Save the model even if CV does not beat the baseline (not recommended).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 55)
    print("  Bug Assignment — LTR Training Script")
    print("=" * 55)
    print(f"  Dataset    : {args.csv_path}")
    print(f"  Test size  : {args.test_size}")
    print(f"  Model out  : {args.model_output}")
    print()

    # 1. Load data
    print("[1/5] Loading dataset...")
    bugs, dev_stats = csv_load(args.csv_path)
    devs, resolutions = build_developers(dev_stats)

    resolvable = [b for b in bugs if b.id in resolutions]
    rng = random.Random(args.seed)
    rng.shuffle(resolvable)

    n_test    = args.test_size
    test_bugs  = resolvable[:n_test]
    train_bugs = resolvable[n_test:]
    print(f"  Train: {len(train_bugs):,} bugs  |  Test (held out): {len(test_bugs):,} bugs")

    # 2. Fit the base assigner on training data
    print("\n[2/5] Fitting BugAssigner on training data...")
    assigner = BugAssigner()
    assigner.fit(train_bugs, devs, resolutions)

    # 3. Build LTR training feature matrix
    print("\n[3/5] Building LTR feature matrix...")
    builder = LTRDataBuilder(assigner)
    X, y, groups, feature_names = builder.build_training_data(train_bugs, resolutions)
    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # 4. Cross-validate (safety gate)
    print("\n[4/5] Running 5-fold cross-validation...")
    trainer = LTRTrainer()
    cv_results = trainer.cross_validate(X, y, groups, n_folds=5)

    passed_gate = cv_results["avg_top1"] > DEFAULT_BASELINE_TOP1
    if not passed_gate:
        print(
            f"\n  [X] CV average Top-1 ({cv_results['avg_top1']:.2f}%) "
            f"did not beat baseline ({DEFAULT_BASELINE_TOP1:.2f}%)."
        )
        if not args.skip_cv_gate:
            print("  Aborting. Use --skip-cv-gate to force save.\n")
            sys.exit(1)
        print("  --skip-cv-gate set — saving anyway.\n")

    # 5. Train on full training set and save
    print("\n[5/5] Training final model on full training set...")
    trainer.train(X, y, groups)
    trainer.save_model(str(args.model_output))
    print(f"\n  [OK] Model saved to {args.model_output}")


if __name__ == "__main__":
    main()

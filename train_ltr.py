"""
train_ltr.py — One-time training script for the LTR model.

Run this AFTER assigner.fit() has been called on the training data.
"""
from pathlib import Path
from src.loaders.csv_loader import load as csv_load
from src.loaders.base import build_developers
from src.models.assigner import BugAssigner
from src.models.ltr_data_builder import LTRDataBuilder
from src.models.ltr_trainer import LTRTrainer

# 1. Load data
CSV_PATH = Path("data") / "eclipse" / "final dataset for work ecllipse.csv"
bugs, dev_stats = csv_load(CSV_PATH)
devs, resolutions = build_developers(dev_stats)

# Only keep bugs that have a known resolution
resolvable = [b for b in bugs if b.id in resolutions]

import random
rng = random.Random(42)
rng.shuffle(resolvable)

# 2. Split (same 98/2 split as test_model.py)
# 200 bugs for testing.
n_test = 200
test_bugs = resolvable[:n_test]
train_bugs = resolvable[n_test:]

# 3. Fit the assigner on training data
assigner = BugAssigner()
assigner.fit(train_bugs, devs, resolutions)

# 4. Build LTR training data
builder = LTRDataBuilder(assigner)
X, y, groups, feature_names = builder.build_training_data(train_bugs, resolutions)

# 5. Cross-validate FIRST (safety gate)
trainer = LTRTrainer()
cv_results = trainer.cross_validate(X, y, groups, n_folds=5)

# 6. If CV passes, train final model
BASELINE_TOP1 = 22.00
if cv_results["avg_top1"] > BASELINE_TOP1:
    trainer.train(X, y, groups)
    trainer.save_model("models/ltr_ranker.json")
    print("✅ Model saved. Proceed to Phase 5.")
else:
    print("❌ CV failed safety gate. Do not proceed.")


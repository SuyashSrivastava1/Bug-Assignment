"""
src/models/ltr_trainer.py

Trains and validates the Learning-to-Rank model using K-Fold cross-validation.
"""

import json
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime


class LTRTrainer:
    """
    Trains an XGBRanker on the bug assignment task with safety validation.
    """

    # Conservative hyperparameters to avoid overfitting
    DEFAULT_PARAMS = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@5",
        "tree_method": "hist",
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    def __init__(self, params: dict = None):
        self.params = params or self.DEFAULT_PARAMS.copy()
        self.model = None
        self.training_metrics = {}

    def train(self, X: np.ndarray, y: np.ndarray, groups: list) -> xgb.XGBRanker:
        """
        Train the ranker on the full training set.

        Parameters
        ----------
        X : Feature matrix (total_candidates, 19)
        y : Labels (total_candidates,)
        groups : List of candidate counts per bug
        """
        self.model = xgb.XGBRanker(**self.params)
        self.model.fit(X, y, group=groups, verbose=False)

        # Record training metadata
        self.training_metrics = {
            "n_bugs": len(groups),
            "n_candidates": len(y),
            "n_positive": int(y.sum()),
            "n_features": X.shape[1],
            "avg_candidates_per_bug": round(np.mean(groups), 1),
            "trained_at": datetime.now().isoformat(),
        }

        return self.model

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: list,
        n_folds: int = 5,
    ) -> dict:
        """
        K-Fold cross-validation at the BUG level (not row level).

        CRITICAL: We split by BUGS (groups), not by individual rows.
        This prevents data leakage where candidates from the same bug
        appear in both train and validation sets.

        Returns
        -------
        dict with per-fold and aggregate metrics
        """
        n_bugs = len(groups)
        bug_indices = np.arange(n_bugs)
        np.random.seed(42)
        np.random.shuffle(bug_indices)

        fold_size = n_bugs // n_folds
        fold_results = []

        # Convert groups to cumulative row offsets
        group_offsets = np.cumsum([0] + groups)

        for fold in range(n_folds):
            # Determine which bugs go into validation
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_bugs
            val_bug_indices = bug_indices[val_start:val_end]
            train_bug_indices = np.concatenate([
                bug_indices[:val_start],
                bug_indices[val_end:]
            ])

            # Expand bug indices to row indices
            train_rows = []
            train_groups = []
            for bi in train_bug_indices:
                start, end = group_offsets[bi], group_offsets[bi + 1]
                train_rows.extend(range(start, end))
                train_groups.append(end - start)

            val_rows = []
            val_groups = []
            for bi in val_bug_indices:
                start, end = group_offsets[bi], group_offsets[bi + 1]
                val_rows.extend(range(start, end))
                val_groups.append(end - start)

            X_train, y_train = X[train_rows], y[train_rows]
            X_val, y_val = X[val_rows], y[val_rows]

            # Train fold model
            fold_model = xgb.XGBRanker(**self.params)
            fold_model.fit(X_train, y_train, group=train_groups, verbose=False)

            # Evaluate: compute Top-1 accuracy on validation bugs
            scores = fold_model.predict(X_val)
            top1_correct = 0
            offset = 0
            for g in val_groups:
                group_scores = scores[offset:offset + g]
                group_labels = y_val[offset:offset + g]
                predicted_best = np.argmax(group_scores)
                if group_labels[predicted_best] == 1:
                    top1_correct += 1
                offset += g

            top1_acc = top1_correct / len(val_groups) * 100
            fold_results.append({
                "fold": fold + 1,
                "n_train_bugs": len(train_bug_indices),
                "n_val_bugs": len(val_bug_indices),
                "top1_accuracy": round(top1_acc, 2),
            })

            print(f"  Fold {fold+1}/{n_folds}: "
                  f"Top-1 = {top1_acc:.2f}% "
                  f"(train={len(train_bug_indices)}, val={len(val_bug_indices)})")

        avg_top1 = np.mean([f["top1_accuracy"] for f in fold_results])
        std_top1 = np.std([f["top1_accuracy"] for f in fold_results])

        cv_results = {
            "n_folds": n_folds,
            "fold_results": fold_results,
            "avg_top1": round(avg_top1, 2),
            "std_top1": round(std_top1, 2),
        }

        print(f"\n  CV Average Top-1: {avg_top1:.2f}% ± {std_top1:.2f}%")

        return cv_results

    def save_model(self, path: str):
        """Save the trained model to a JSON file."""
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))

        # Save metadata alongside the model
        meta_path = model_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(self.training_metrics, f, indent=2)

        print(f"  Model saved to {model_path}")
        print(f"  Metadata saved to {meta_path}")

    def load_model(self, path: str):
        """Load a previously trained model."""
        self.model = xgb.XGBRanker(**self.params)
        self.model.load_model(path)
        return self.model

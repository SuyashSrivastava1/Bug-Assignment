# Learning-to-Rank (LTR) Implementation Guide

## Bug-to-Developer Assignment System — Re-Ranking Upgrade

> [!IMPORTANT]
> This guide is designed to be followed **sequentially**. Each phase has a **checkpoint** at the end.
> Do NOT proceed to the next phase until the checkpoint passes.

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Prerequisites](#2-prerequisites)
3. [Safety Philosophy](#3-safety-philosophy)
4. [Phase 1 — Baseline Snapshot](#phase-1--baseline-snapshot)
5. [Phase 2 — Training Data Construction](#phase-2--training-data-construction)
6. [Phase 3 — Feature Engineering](#phase-3--feature-engineering)
7. [Phase 4 — Model Training & Validation](#phase-4--model-training--validation)
8. [Phase 5 — Integration into the Pipeline](#phase-5--integration-into-the-pipeline)
9. [Phase 6 — Safety Guardrails](#phase-6--safety-guardrails)
10. [Phase 7 — Evaluation & Comparison](#phase-7--evaluation--comparison)
11. [Phase 8 — Rollback Procedure](#phase-8--rollback-procedure)
12. [Appendix A — Feature Reference Table](#appendix-a--feature-reference-table)
13. [Appendix B — Hyperparameter Tuning Guide](#appendix-b--hyperparameter-tuning-guide)
14. [Appendix C — Troubleshooting](#appendix-c--troubleshooting)

---

## 1. Overview & Architecture

### What changes

```
CURRENT PIPELINE (semantic-prototype-weights branch):
┌─────────────────────────────────────────────────────────────┐
│  Step 1-2: Hybrid Search (Dense + TF-IDF) → top-K bugs     │  UNCHANGED
│  Step 3:   Candidate Extraction → devs who fixed those bugs │  UNCHANGED
│  Step 4:   Hard Filtering → remove on-leave / overloaded    │  UNCHANGED
│  Step 5:   KNN Affinity → per-dev expertise score           │  UNCHANGED
│  Step 6:   Build 6-column attribute matrix                  │  EXPANDED (more features)
│  Step 7:   Semantic Prototype Weights                       │  REPLACED by LTR
│  Step 8:   WSM linear scoring + component bonus             │  REPLACED by LTR
└─────────────────────────────────────────────────────────────┘

NEW PIPELINE:
┌─────────────────────────────────────────────────────────────┐
│  Step 1-6: Exactly the same                                 │
│  Step 7:   Build extended feature matrix (N features/dev)   │  NEW
│  Step 8:   LTR model predicts relevance score per candidate │  NEW
│  Step 9:   Confidence check → fallback to WSM if uncertain  │  NEW (safety)
└─────────────────────────────────────────────────────────────┘
```

### What does NOT change

- Bug and Developer data models (`bug.py`, `developer.py`)
- Data loaders (`csv_loader.py`, `bugzilla_loader.py`)
- The hybrid search logic (Steps 1-2)
- KNN Affinity computation (Step 5)
- The test harness (`test_model.py`)
- The web UI / router (`router.py`, `main.py`)

---

## 2. Prerequisites

### Python Packages

```
# Add to requirements.txt
xgboost>=2.0.0
shap>=0.43.0        # for explainability (optional but recommended)
```

### Knowledge Requirements

Before proceeding, ensure you understand:

- [x] How the current `assigner.py` pipeline works (Steps 1-8)
- [x] What the 6-column attribute matrix contains
- [x] How the WSM (Weighted Sum Model) scores candidates
- [x] What Top-1/3/5 accuracy and MRR mean
- [x] Basic understanding of decision trees / gradient boosting

### Hardware Requirements

- Training the LTR model on 10,000 bugs: **< 30 seconds** on any modern CPU
- No GPU needed — XGBoost runs efficiently on CPU for this dataset size
- RAM: ~500 MB additional during training

---

## 3. Safety Philosophy

> [!CAUTION]
> The LTR model MUST NOT silently degrade performance. Every step below includes
> validation gates. If any gate fails, the system falls back to the existing WSM.

### Core Safety Rules

| Rule # | Rule | Rationale |
|:---:|:---|:---|
| S1 | **Never delete the WSM code** | It is your fallback. LTR wraps around it. |
| S2 | **Always train with cross-validation** | Prevents overfitting to one train/test split |
| S3 | **Confidence threshold on LTR output** | If the model is uncertain, WSM takes over |
| S4 | **A/B flag in the assigner** | Switch between LTR and WSM with a single boolean |
| S5 | **Log both scores** | Always compute WSM AND LTR scores, log both, return the active one |
| S6 | **Never train on test data** | The 98/2 split must be respected. LTR trains on the 98% only. |
| S7 | **Version the trained model** | Save the `.json` model file with a timestamp so you can rollback |

---

## Phase 1 — Baseline Snapshot

### Purpose
Lock in your current performance numbers so you have a clear comparison target.

### Steps

#### 1.1 Create a new branch

```powershell
git checkout semantic-prototype-weights
git checkout -b learning-to-rank
```

#### 1.2 Run the full test suite and save output

```powershell
python test_model.py --quick > baseline_results.txt 2>&1
```

#### 1.3 Record baseline metrics

Create a file `metrics_log.md` in the project root:

```markdown
# Metrics Log

## Baseline (semantic-prototype-weights)
- Top-1: 22.00%
- Top-3: 49.50%
- Top-5: 65.00%
- MRR:   0.3954
- Latency: 59.3 ms/bug
- Date: YYYY-MM-DD
```

#### 1.4 Commit the baseline

```powershell
git add baseline_results.txt metrics_log.md
git commit -m "chore: record baseline metrics before LTR implementation"
```

### ✅ Checkpoint 1
- [ ] New branch `learning-to-rank` exists
- [ ] Baseline metrics recorded in `metrics_log.md`
- [ ] All 67 tests pass on this branch
- [ ] No uncommitted changes

---

## Phase 2 — Training Data Construction

### Purpose
Build the labeled dataset that the LTR model will learn from.

### 2.1 Understand the data structure

For Learning-to-Rank, you need **query-document pairs with relevance labels**:

```
Query   = a bug (to be assigned)
Document = a candidate developer
Label   = 1 if this developer actually fixed this bug, 0 otherwise
Group   = how many candidates belong to this query (bug)
```

### 2.2 Create `src/models/ltr_data_builder.py`

This module constructs training data by replaying the assignment pipeline on historical bugs.

```python
"""
src/models/ltr_data_builder.py

Builds training data for the Learning-to-Rank model by replaying the
assignment pipeline on historical bugs with known resolutions.
"""

import numpy as np
from src.models.bug import Bug
from src.models.developer import Developer


class LTRDataBuilder:
    """
    Replays the candidate extraction pipeline for each training bug
    and builds (features, labels, groups) suitable for XGBRanker.
    """

    def __init__(self, assigner):
        """
        Parameters
        ----------
        assigner : BugAssigner (already fit)
            The trained assigner whose pipeline we replay.
        """
        self.assigner = assigner

    def build_training_data(
        self,
        train_bugs: list,
        resolutions: dict,
        k: int = 20,
    ) -> tuple:
        """
        Replay the pipeline for each bug and collect feature rows.

        Parameters
        ----------
        train_bugs : list of Bug — training split only
        resolutions : dict — bug_id -> developer_id
        k : int — number of KNN neighbors (same as assign())

        Returns
        -------
        X : np.ndarray — shape (total_candidates, num_features)
        y : np.ndarray — shape (total_candidates,) — 1 for correct dev, 0 otherwise
        groups : list[int] — number of candidates per bug (for XGBRanker)
        feature_names : list[str] — column names for interpretability
        """
        all_features = []
        all_labels = []
        groups = []
        skipped = 0

        for bug in train_bugs:
            actual_dev_id = resolutions.get(bug.id)
            if actual_dev_id is None:
                skipped += 1
                continue

            # Get candidate features using the assigner's internal pipeline
            # (we'll add a method to BugAssigner for this)
            candidate_data = self.assigner._get_candidate_features(bug, k=k)

            if candidate_data is None or len(candidate_data) == 0:
                skipped += 1
                continue

            bug_features = []
            bug_labels = []

            for dev, features in candidate_data:
                bug_features.append(features)
                label = 1 if dev.id == actual_dev_id else 0
                bug_labels.append(label)

            # Only include this bug if the correct developer is in the candidate pool
            if sum(bug_labels) == 0:
                skipped += 1
                continue

            all_features.extend(bug_features)
            all_labels.extend(bug_labels)
            groups.append(len(bug_features))

        print(f"[LTR Data] Built {len(all_features)} rows from "
              f"{len(groups)} bugs (skipped {skipped})")

        X = np.array(all_features, dtype=float)
        y = np.array(all_labels, dtype=float)

        feature_names = self._get_feature_names()

        return X, y, groups, feature_names

    def _get_feature_names(self) -> list:
        """Return ordered feature column names."""
        return [
            # Original 6 attributes
            "experience",
            "fix_time",
            "success_rate",
            "workload",
            "domain_skill",
            "knn_affinity",
            # Extended features (Phase 3)
            "component_match",
            "severity_is_critical",
            "severity_is_normal",
            "severity_is_minor",
            "num_past_bugs_fixed",
            "cosine_sim_max",
            "cosine_sim_mean",
            "prototype_experience",
            "prototype_fix_time",
            "prototype_success_rate",
            "prototype_workload",
            "prototype_domain_skill",
            "prototype_knn_affinity",
        ]
```

### 2.3 What `_get_candidate_features()` does

You'll need to add a new method to `BugAssigner` that runs Steps 1-6 of the pipeline
and returns raw feature vectors per candidate (without scoring). This is detailed in Phase 5.

### ✅ Checkpoint 2
- [ ] `ltr_data_builder.py` created
- [ ] You understand the query/document/label/group structure
- [ ] No code in `assigner.py` has been modified yet
- [ ] All 67 tests still pass

---

## Phase 3 — Feature Engineering

### Purpose
Design the feature set that the LTR model will use to rank candidates. More features = more
signal for the model to learn from.

### 3.1 Feature Categories

#### Category A — Original Developer Attributes (already exist)

| # | Feature | Type | Source | Description |
|:-:|:---|:---:|:---|:---|
| 1 | `experience` | Benefit | `dev.experience` | Developer's overall experience score |
| 2 | `fix_time` | Cost | `dev.fix_time` | Average time to fix bugs (lower = better) |
| 3 | `success_rate` | Benefit | `dev.success_rate` | Fraction of bugs fixed successfully |
| 4 | `workload` | Cost | `dev.workload` | Current workload (lower = more available) |
| 5 | `domain_skill` | Benefit | `dev.domain_skill` | Domain expertise score |
| 6 | `knn_affinity` | Benefit | Computed | Mean cosine similarity to dev's past bugs |

#### Category B — Bug-Developer Interaction Features (NEW)

| # | Feature | Type | Source | Description |
|:-:|:---|:---:|:---|:---|
| 7 | `component_match` | Binary | Computed | 1 if dev has fixed bugs in this module before |
| 8 | `num_past_bugs_fixed` | Count | `_dev_bug_indices` | Total number of bugs this dev has fixed |
| 9 | `cosine_sim_max` | Float | Computed | Max cosine similarity between new bug and this dev's past bugs |
| 10 | `cosine_sim_mean` | Float | Computed | Mean cosine similarity (same as knn_affinity but included explicitly) |

#### Category C — Bug Context Features (NEW)

| # | Feature | Type | Source | Description |
|:-:|:---|:---:|:---|:---|
| 11 | `severity_is_critical` | Binary | `bug.severity` | 1 if severity == "critical" |
| 12 | `severity_is_normal` | Binary | `bug.severity` | 1 if severity == "normal" |
| 13 | `severity_is_minor` | Binary | `bug.severity` | 1 if severity == "minor" |

#### Category D — Semantic Prototype Scores (NEW — keep your existing innovation!)

| # | Feature | Type | Source | Description |
|:-:|:---|:---:|:---|:---|
| 14 | `prototype_experience` | Float | `_compute_semantic_weights` | Cosine sim to Experience prototype |
| 15 | `prototype_fix_time` | Float | `_compute_semantic_weights` | Cosine sim to Fix Time prototype |
| 16 | `prototype_success_rate` | Float | `_compute_semantic_weights` | Cosine sim to Success Rate prototype |
| 17 | `prototype_workload` | Float | `_compute_semantic_weights` | Cosine sim to Workload prototype |
| 18 | `prototype_domain_skill` | Float | `_compute_semantic_weights` | Cosine sim to Domain Skill prototype |
| 19 | `prototype_knn_affinity` | Float | `_compute_semantic_weights` | Cosine sim to KNN Affinity prototype |

> [!TIP]
> The prototype scores from Step 7 become **features** for the LTR model instead of being
> directly used as weights. This lets the model learn *how much* each prototype matters,
> rather than you hardcoding the formula.

### 3.2 Total Feature Count: **19 features per candidate**

### 3.3 Feature Construction Code

Add this to `_get_candidate_features()` in `assigner.py` (Phase 5 shows the full method):

```python
# For each available developer, build a 19-element feature vector:
features = [
    # Category A: Developer attributes (6)
    dev.experience,
    dev.fix_time,
    dev.success_rate,
    dev.workload,
    dev.domain_skill,
    knn_affinity_score,

    # Category B: Bug-Developer interaction (4)
    1.0 if bug.module in self._dev_components.get(dev.id, set()) else 0.0,
    len(self._dev_bug_indices.get(dev.id, [])),
    float(similarities[dev_indices].max()) if dev_indices else 0.0,
    float(similarities[dev_indices].mean()) if dev_indices else 0.0,

    # Category C: Bug context (3)
    1.0 if bug.severity == "critical" else 0.0,
    1.0 if bug.severity == "normal" else 0.0,
    1.0 if bug.severity == "minor" else 0.0,

    # Category D: Prototype scores (6) — computed once per bug, same for all devs
    *prototype_scores,
]
```

### ✅ Checkpoint 3
- [ ] Feature list finalized (19 features)
- [ ] You understand which features are per-developer vs per-bug
- [ ] No code modified yet — this is still planning
- [ ] Feature names list matches the order in the construction code

---

## Phase 4 — Model Training & Validation

### Purpose
Train the XGBoost Ranker with proper cross-validation to ensure it generalizes.

### 4.1 Create `src/models/ltr_trainer.py`

```python
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
```

### 4.2 Cross-Validation Safety Gate

> [!WARNING]
> **SAFETY GATE**: The LTR model must beat WSM in cross-validation before you integrate it.
> If the average CV Top-1 accuracy is NOT higher than the WSM baseline (22.00%), STOP.
> Go back to Phase 3 and improve your features.

```python
# Safety check after cross-validation
BASELINE_TOP1 = 22.00  # from semantic-prototype-weights

if cv_results["avg_top1"] <= BASELINE_TOP1:
    print("⚠️  SAFETY GATE FAILED: LTR does not beat WSM baseline.")
    print(f"    LTR CV Top-1: {cv_results['avg_top1']:.2f}%")
    print(f"    WSM Baseline:  {BASELINE_TOP1:.2f}%")
    print("    → Do NOT proceed to integration. Improve features first.")
else:
    print("✅ SAFETY GATE PASSED: LTR beats WSM baseline.")
    print(f"    Improvement: +{cv_results['avg_top1'] - BASELINE_TOP1:.2f}pp")
```

### 4.3 Training Workflow

```powershell
# Run this script to train and validate (create as train_ltr.py in project root)
python train_ltr.py
```

```python
"""
train_ltr.py — One-time training script for the LTR model.

Run this AFTER assigner.fit() has been called on the training data.
"""
from src.loaders.csv_loader import load_eclipse_csv
from src.models.assigner import BugAssigner
from src.models.ltr_data_builder import LTRDataBuilder
from src.models.ltr_trainer import LTRTrainer

# 1. Load data
bugs, devs = load_eclipse_csv("final dataset for work ecllipse.csv")
resolutions = {b.id: ... for b in bugs}  # same as test_model.py

# 2. Split (same 98/2 split as test_model.py)
train_bugs = bugs[:9800]
test_bugs = bugs[9800:]

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
```

### ✅ Checkpoint 4
- [ ] `ltr_trainer.py` created with `train()`, `cross_validate()`, `save_model()`, `load_model()`
- [ ] `ltr_data_builder.py` created
- [ ] Cross-validation runs and reports per-fold Top-1 accuracy
- [ ] **Safety gate passed**: CV Top-1 > 22.00% (WSM baseline)
- [ ] Model saved to `models/ltr_ranker.json`
- [ ] All 67 existing tests still pass (no regressions)

---

## Phase 5 — Integration into the Pipeline

### Purpose
Add the LTR model to `BugAssigner` with an A/B switch and confidence fallback.

### 5.1 Add `_get_candidate_features()` to `BugAssigner`

This is the bridge between the existing pipeline and the LTR model. It runs Steps 1-6
and returns raw feature vectors instead of scores.

```python
def _get_candidate_features(self, bug: Bug, k: int = 20) -> list | None:
    """
    Run Steps 1-6 of the pipeline and return raw feature vectors
    per candidate developer. Used by LTR training and inference.

    Returns
    -------
    list of (Developer, feature_vector) tuples, or None if no candidates.
    """
    if self._historical_dense is None:
        raise RuntimeError("BugAssigner not trained. Call fit() first.")

    # Step 1 — Feature Extraction
    new_dense = self._dense_model.encode([bug.description])
    new_sparse = self._sparse_model.transform([bug.description])

    # Step 2 — Hybrid Similarity
    sim_dense = cosine_similarity(new_dense, self._historical_dense)[0]
    sim_sparse = cosine_similarity(new_sparse, self._historical_sparse)[0]
    similarities = (sim_dense * 0.5) + (sim_sparse * 0.5)
    top_k_indices = similarities.argsort()[-k:][::-1]

    # Step 3 — Candidate Extraction
    candidate_ids = set()
    for idx in top_k_indices:
        b = self._historical_bugs[idx]
        if b.id in self._resolutions:
            candidate_ids.add(self._resolutions[b.id])

    # Step 4 — Hard Filtering
    available = [
        d for d in self._developers
        if d.id in candidate_ids and not d.on_leave and d.workload < 0.95
    ]

    if not available:
        return None

    # Step 5 — KNN Affinity
    knn_affinity = []
    for d in available:
        indices = self._dev_bug_indices.get(d.id, [])
        if indices:
            knn_affinity.append(float(similarities[indices].mean()))
        else:
            knn_affinity.append(0.0)

    # Prototype scores (computed once, same for all candidates)
    text = bug.description + " " + bug.severity
    bug_vec = self._dense_model.encode([text])
    raw_sims = cosine_similarity(bug_vec, self._prototype_vectors)[0]
    prototype_scores = list((raw_sims + 1.0) / 2.0)  # map to [0, 1]

    # Build feature vectors
    result = []
    for i, dev in enumerate(available):
        dev_indices = self._dev_bug_indices.get(dev.id, [])

        features = [
            # Category A: Developer attributes (6)
            dev.experience,
            dev.fix_time,
            dev.success_rate,
            dev.workload,
            dev.domain_skill,
            knn_affinity[i],

            # Category B: Bug-Developer interaction (4)
            1.0 if bug.module in self._dev_components.get(dev.id, set()) else 0.0,
            float(len(dev_indices)),
            float(similarities[dev_indices].max()) if dev_indices else 0.0,
            float(similarities[dev_indices].mean()) if dev_indices else 0.0,

            # Category C: Bug context (3)
            1.0 if bug.severity == "critical" else 0.0,
            1.0 if bug.severity == "normal" else 0.0,
            1.0 if bug.severity == "minor" else 0.0,

            # Category D: Prototype scores (6)
            *prototype_scores,
        ]

        result.append((dev, features))

    return result
```

### 5.2 Modify `assign()` with A/B switch

```python
def __init__(self):
    # ... existing init code ...
    self._ltr_model = None          # XGBRanker, loaded if available
    self._use_ltr = False           # A/B switch
    self._ltr_confidence_min = 0.1  # minimum score gap to trust LTR

def load_ltr_model(self, path: str):
    """Load a trained LTR model. Call after fit()."""
    import xgboost as xgb
    self._ltr_model = xgb.XGBRanker()
    self._ltr_model.load_model(path)
    self._use_ltr = True
    print(f"[LTR] Model loaded from {path}")

def assign(self, new_bug: Bug, k: int = 20) -> tuple:
    # ... Steps 1-6 remain exactly the same ...

    if self._use_ltr and self._ltr_model is not None:
        # ---- LTR PATH ----
        return self._assign_ltr(new_bug, available, similarities, k)
    else:
        # ---- WSM PATH (existing code, untouched) ----
        return self._assign_wsm(new_bug, available, similarities, matrix, k)
```

### 5.3 Add `_assign_ltr()` method

```python
def _assign_ltr(self, bug, available, similarities, k):
    """Score candidates using the trained LTR model with confidence fallback."""

    candidate_data = self._get_candidate_features(bug, k)
    if candidate_data is None:
        return None, [], None, {}

    devs = [d for d, _ in candidate_data]
    X = np.array([f for _, f in candidate_data], dtype=float)

    # LTR prediction
    scores = self._ltr_model.predict(X)

    # ---- CONFIDENCE CHECK (Safety Rule S3) ----
    sorted_scores = np.sort(scores)[::-1]
    if len(sorted_scores) >= 2:
        score_gap = sorted_scores[0] - sorted_scores[1]
        if score_gap < self._ltr_confidence_min:
            # Model is not confident → fall back to WSM
            return self._assign_wsm_fallback(bug, available, similarities)

    # Rank by LTR scores
    order = scores.argsort()[::-1]
    ranked = [(devs[i], float(scores[i])) for i in order]

    # Signals for debugging/display
    weights, signals = self._compute_semantic_weights(bug)
    signals["ranking_method"] = "ltr"
    signals["ltr_confidence_gap"] = round(float(score_gap), 4) if len(sorted_scores) >= 2 else None

    return ranked[0][0], ranked, weights, signals
```

### 5.4 File structure after integration

```
src/models/
├── __init__.py
├── assigner.py           # Modified: added _get_candidate_features, _assign_ltr, A/B switch
├── bug.py                # UNCHANGED
├── developer.py          # UNCHANGED
├── ltr_data_builder.py   # NEW
└── ltr_trainer.py        # NEW

models/
├── ltr_ranker.json       # NEW: trained XGBoost model
└── ltr_ranker.meta.json  # NEW: training metadata
```

### ✅ Checkpoint 5
- [ ] `_get_candidate_features()` added to `BugAssigner`
- [ ] `_assign_ltr()` added with confidence fallback
- [ ] A/B switch (`_use_ltr`) works: `False` → WSM path, `True` → LTR path
- [ ] With `_use_ltr = False`, all 67 tests still pass (WSM unchanged)
- [ ] With `_use_ltr = True`, `assign()` returns valid results

---

## Phase 6 — Safety Guardrails

### 6.1 Dual Logging

Always compute and log both WSM and LTR scores, regardless of which is active:

```python
# In assign(), always log both paths for comparison
wsm_result = self._assign_wsm(...)
ltr_result = self._assign_ltr(...)

# Log disagreements
if wsm_result[0] != ltr_result[0]:
    print(f"[DISAGREE] Bug {bug.id}: WSM→{wsm_result[0].id}, LTR→{ltr_result[0].id}")
```

> [!NOTE]
> This dual logging is for development/testing only. In production, you'd remove it
> for performance reasons and only run the active path.

### 6.2 Confidence Threshold Tuning

The `_ltr_confidence_min` parameter controls when LTR falls back to WSM:

| Value | Behavior | When to use |
|:---:|:---|:---|
| `0.0` | Never fall back (always trust LTR) | Only after extensive validation |
| `0.05` | Fall back when top-2 are very close | Moderate safety |
| `0.1` | Fall back more aggressively | **Recommended starting value** |
| `0.2` | Fall back frequently | Very conservative / initial testing |

### 6.3 Kill Switch

Add an environment variable override so you can disable LTR without code changes:

```python
import os

def assign(self, new_bug, k=20):
    use_ltr = self._use_ltr and os.environ.get("DISABLE_LTR") != "1"
    # ...
```

```powershell
# Emergency disable in production:
$env:DISABLE_LTR = "1"
python main.py
```

### 6.4 Model Versioning

```
models/
├── ltr_ranker_v1_2026-05-24.json      # First version
├── ltr_ranker_v1_2026-05-24.meta.json
├── ltr_ranker_v2_2026-05-30.json      # After tuning
├── ltr_ranker_v2_2026-05-30.meta.json
└── ltr_ranker.json → ltr_ranker_v2_2026-05-30.json  # symlink to active
```

### ✅ Checkpoint 6
- [ ] Confidence fallback implemented and tested
- [ ] Kill switch via environment variable works
- [ ] Model file is versioned with timestamp
- [ ] Dual logging works during development

---

## Phase 7 — Evaluation & Comparison

### 7.1 Run the standard test suite with LTR

```powershell
python test_model.py --quick
```

### 7.2 Run a comparison test

Create `compare_models.py` to run both WSM and LTR on the same test set:

```python
"""
compare_models.py — Side-by-side comparison of WSM vs LTR.
"""

# For each test bug:
#   1. Run assign() with _use_ltr = False → WSM result
#   2. Run assign() with _use_ltr = True  → LTR result
#   3. Compare Top-1/3/5 accuracy and MRR for both

# Output:
#   ┌─────────┬────────┬────────┐
#   │ Metric  │  WSM   │  LTR   │
#   ├─────────┼────────┼────────┤
#   │ Top-1   │ 22.00% │ ??%    │
#   │ Top-3   │ 49.50% │ ??%    │
#   │ Top-5   │ 65.00% │ ??%    │
#   │ MRR     │ 0.3954 │ ??     │
#   └─────────┴────────┴────────┘
```

### 7.3 Expected Results & Decision Matrix

| Outcome | Action |
|:---|:---|
| LTR beats WSM on all metrics | ✅ Deploy LTR as default, keep WSM as fallback |
| LTR beats WSM on Top-1 but not Top-5 | ⚠️ Investigate. Consider ensemble. |
| LTR ties with WSM | ⚠️ Not worth the complexity. Stay with WSM. |
| LTR is worse than WSM | ❌ Do NOT deploy. Go back to Phase 3 (features). |

### 7.4 Feature Importance Analysis

After training, inspect which features the model relies on most:

```python
import xgboost as xgb

model = xgb.XGBRanker()
model.load_model("models/ltr_ranker.json")

importance = model.get_booster().get_score(importance_type="gain")
for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"  {feat}: {score:.2f}")
```

> [!TIP]
> If `knn_affinity` and `cosine_sim_max` dominate, it means the model is learning that
> topical similarity is the strongest signal — which validates your existing architecture.
> If `prototype_*` features are highly ranked, it means the semantic weighting approach
> is valuable and the model is learning to use it non-linearly.

### 7.5 Update metrics_log.md

```markdown
## LTR v1 (learning-to-rank branch)
- Top-1: ??%
- Top-3: ??%
- Top-5: ??%
- MRR:   ??
- Latency: ?? ms/bug
- CV Top-1: ??% ± ??%
- Date: YYYY-MM-DD
- Safety gate: PASSED / FAILED
```

### ✅ Checkpoint 7
- [ ] `compare_models.py` run and results recorded
- [ ] Feature importance analysis completed
- [ ] `metrics_log.md` updated with LTR results
- [ ] Decision made: deploy / do not deploy

---

## Phase 8 — Rollback Procedure

### If LTR needs to be disabled after deployment

#### Option A: Environment Variable (instant, no code change)

```powershell
$env:DISABLE_LTR = "1"
# Restart the application
```

#### Option B: Code Change (permanent)

```python
# In assigner.py __init__():
self._use_ltr = False  # Disable LTR, revert to WSM
```

#### Option C: Branch Rollback (nuclear option)

```powershell
git checkout semantic-prototype-weights
# You're back to the previous working version with zero LTR code
```

#### Rollback Checklist

- [ ] Set `DISABLE_LTR=1` or `_use_ltr = False`
- [ ] Run `python test_model.py --quick` to confirm all 67 tests pass
- [ ] Verify Top-1/3/5 and MRR match the baseline in `metrics_log.md`
- [ ] Notify stakeholders that LTR has been disabled
- [ ] Create a ticket to investigate what went wrong

---

## Appendix A — Feature Reference Table

| # | Feature Name | Dimension | Type | Range | Description |
|:-:|:---|:---:|:---:|:---:|:---|
| 1 | `experience` | Per-dev | Benefit | [0, 1] | Overall experience score |
| 2 | `fix_time` | Per-dev | Cost | [0, 1] | Avg fix time (lower better) |
| 3 | `success_rate` | Per-dev | Benefit | [0, 1] | Fix success ratio |
| 4 | `workload` | Per-dev | Cost | [0, 1] | Current load (lower better) |
| 5 | `domain_skill` | Per-dev | Benefit | [0, 1] | Domain expertise |
| 6 | `knn_affinity` | Per-dev × Per-bug | Benefit | [0, 1] | Mean cosine sim to dev's bugs |
| 7 | `component_match` | Per-dev × Per-bug | Binary | {0, 1} | Module overlap |
| 8 | `num_past_bugs_fixed` | Per-dev | Count | [0, ∞) | Total bugs fixed |
| 9 | `cosine_sim_max` | Per-dev × Per-bug | Benefit | [-1, 1] | Max similarity to dev's bugs |
| 10 | `cosine_sim_mean` | Per-dev × Per-bug | Benefit | [-1, 1] | Mean similarity to dev's bugs |
| 11 | `severity_is_critical` | Per-bug | Binary | {0, 1} | Bug is critical? |
| 12 | `severity_is_normal` | Per-bug | Binary | {0, 1} | Bug is normal? |
| 13 | `severity_is_minor` | Per-bug | Binary | {0, 1} | Bug is minor? |
| 14 | `prototype_experience` | Per-bug | Float | [0, 1] | Semantic sim to Experience prototype |
| 15 | `prototype_fix_time` | Per-bug | Float | [0, 1] | Semantic sim to Fix Time prototype |
| 16 | `prototype_success_rate` | Per-bug | Float | [0, 1] | Semantic sim to Success Rate prototype |
| 17 | `prototype_workload` | Per-bug | Float | [0, 1] | Semantic sim to Workload prototype |
| 18 | `prototype_domain_skill` | Per-bug | Float | [0, 1] | Semantic sim to Domain Skill prototype |
| 19 | `prototype_knn_affinity` | Per-bug | Float | [0, 1] | Semantic sim to KNN Affinity prototype |

---

## Appendix B — Hyperparameter Tuning Guide

Only tune AFTER the basic model works and passes the safety gate.

| Parameter | Default | Tune Range | Effect |
|:---|:---:|:---:|:---|
| `n_estimators` | 150 | 50–500 | More trees = more expressive, risk overfitting |
| `max_depth` | 4 | 2–8 | Deeper = more complex interactions |
| `learning_rate` | 0.1 | 0.01–0.3 | Lower = slower training, often better results |
| `subsample` | 0.8 | 0.5–1.0 | Row sampling per tree (regularization) |
| `colsample_bytree` | 0.8 | 0.5–1.0 | Feature sampling per tree (regularization) |
| `min_child_weight` | 5 | 1–20 | Higher = more conservative splits |
| `reg_alpha` | 0.1 | 0–1.0 | L1 regularization |
| `reg_lambda` | 1.0 | 0.1–10.0 | L2 regularization |

> [!WARNING]
> Always re-run 5-fold cross-validation after changing hyperparameters.
> Never tune on the test set (the 200 bugs in `test_model.py`).

---

## Appendix C — Troubleshooting

| Problem | Likely Cause | Fix |
|:---|:---|:---|
| CV Top-1 < baseline | Features too noisy or model overfitting | Reduce `max_depth` to 2-3, increase `min_child_weight` |
| LTR always falls back to WSM | `_ltr_confidence_min` too high | Lower it to 0.05 or 0.0 |
| Training data has 0 positive labels | Correct dev not in candidate pool (K too small) | Increase K to 30-40 during training data construction |
| `_get_candidate_features` returns None | All candidates filtered out | Check hard filtering thresholds |
| Feature importance shows only 2-3 features matter | Other features are redundant | Not necessarily a problem — simpler model is better |
| Test accuracy drops after LTR | Data leakage or different train/test split | Verify the split matches `test_model.py` exactly |
| XGBoost import error | Package not installed | `pip install xgboost>=2.0.0` |

# Bug-to-Developer Assignment System

An AI-driven framework that automatically assigns software bug reports to the most suitable developer, using a hybrid pipeline of **NLP text analysis** (Dense/Sparse Embeddings + KNN) and a **Learning-to-Rank (LTR) XGBoost Model**.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

You will be prompted to choose a mode:

| Mode | What you provide | What it does |
|---|---|---|
| **1 — GitHub URL** | A GitHub issue URL | Fetches the bug + the repo's full contributor history, then assigns the bug to the best developer from that project |
| **2 — Custom description** | Plain text | Uses the offline Eclipse + Bugzilla datasets as NLP training context and assigns the bug to the closest matching Eclipse developer |

---

## How the Algorithm Works

### The Pipeline

```
New Bug Report
      │
      ▼
1. Dense+Sparse Embeddings ──► Convert bug text into a hybrid dense+sparse feature vector using `all-MiniLM-L6-v2` and TF-IDF
      │
      ▼
2. KNN Similarity Search   ──► Find the top-K most similar historical bugs using hybrid cosine similarity
      │
      ▼
3. Candidate Extraction    ──► Identify the developers who fixed those similar bugs
      │
      ▼
4. Hard Filtering          ──► Remove developers who are on leave or over 95% workload
      │
      ▼
5. Feature Engineering     ──► Build a matrix of 19 features (candidate metrics, component match, severity, prototype semantic scores)
      │
      ▼
6. Learning-To-Rank        ──► Pass features into an XGBoost Ranker (`objective: rank:ndcg`) to score each candidate
      │
      ▼
7. Final Assignment        ──► Return the highest scored candidate from XGBoost
```

### The Feature Space (19 Features)

The XGBoost Ranker uses 19 distinct features to evaluate each candidate:

1. **Category A: Candidate Static Metrics (6)**
   * **Experience**: Total bugs historically fixed
   * **Fix Time**: Average hours to close a bug
   * **Success Rate**: % of bugs not re-opened
   * **Workload**: Current open issue load (0–100%)
   * **Domain Skill**: Number of distinct components worked on
   * **KNN Affinity**: Mean cosine similarity between the current query bug and all past bugs this candidate has fixed

2. **Category B: Developer-Bug Interaction (4)**
   * **Component Match**: Boolean (1.0/0.0) if the candidate previously worked on this bug's module
   * **Num Past Bugs Fixed**: Count of past bugs fixed (excluding the current one during training)
   * **Cosine Sim Max**: Max similarity score to the candidate's past bugs
   * **Cosine Sim Mean**: Mean similarity score to the candidate's past bugs

3. **Category C: Bug Context (3)**
   * **Severity Flags**: Binary flags for `is_critical`, `is_normal`, `is_minor`

4. **Category D: Semantic Prototype Scores (6)**
   * Continuous NLP signals computed by comparing the bug's dense embedding to 6 predefined semantic "prototype" sentences (e.g. *critical production crash*, *complex algorithm memory leak*, *routine UI alignment*). These act as a continuous semantic hash that helps the XGBoost model learn non-linear relationships based on bug context.

---

## 📈 Evaluation & Research Results

Throughout the development of this assignment system, we iterated through 5 major architectural approaches. Each was rigorously tested against a standardized hold-out test set of **2,000 Eclipse bugs** against a training index of **8,000 historical bugs** and **209 unique developers**. 

The **Learning-to-Rank** model successfully generalized the patterns and definitively dominated all previous heuristic and semantic approaches!

| Metric | `main` (Baseline) | `accuracy-improvements` | `hybrid-search` | `semantic-prototype-weights` | `learning-to-rank` (XGBoost) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | 22.50% | 22.50% | 23.00% | 22.00% | **26.00%** 🏆 |
| **Top-3 Accuracy** | 51.50% | 48.50% | 47.50% | 49.50% | **59.50%** 🏆 |
| **Top-5 Accuracy** | 55.50% | 63.50% | 63.00% | 65.00% | **74.85%** 🏆 |
| **MRR** | 0.3614 | 0.3894 | 0.3974 | 0.3954 | **0.4437** 🏆 |

### Data Leakage Discovery & Fix
During the development of the Learning-to-Rank pipeline, cross-validation metrics initially spiked to an impossible `99.68%`. This was caused by self-retrieval data leakage: the historical feature builder allowed training bugs to query themselves, artificially inflating the `cosine_sim_max` feature to `1.0`. Masking the bug's own index during the similarity search and candidate aggregation successfully removed this leak, resulting in the clean, highly-generalized **26.00%** real-world Top-1 test accuracy shown above.

Run the evaluation yourself:

```bash
python test_model.py           # Full evaluation (~2,000 test bugs)
python test_model.py --quick   # Fast evaluation (200 test bugs)
python test_model.py --unit-only  # Unit + edge case tests only
```

### Training the LTR Model

To retrain the XGBoost ranker on the dataset:
```bash
python train_ltr.py
```
This script runs a pipeline that replays the hybrid search over the training bugs, extracts the 19 features for the candidates, masks the queries to prevent data leakage, runs a 5-Fold Cross Validation, and saves the final production model to `models/ltr_ranker.json`.

---

## Project Structure

```text
Bug Classification/
│
├── src/                        # All source code
│   ├── models/
│   │   ├── bug.py              # Bug data class
│   │   ├── developer.py        # Developer data class
│   │   ├── assigner.py         # Core algorithm pipeline
│   │   ├── ltr_data_builder.py # Extracts LTR feature matrices
│   │   └── ltr_trainer.py      # XGBoost training & CV logic
│   │
│   ├── loaders/
│   │   ├── base.py             # Shared utilities
│   │   ├── csv_loader.py       # Load from Eclipse CSV dataset
│   │   ├── bugzilla_loader.py  # Load from Bugzilla corpus .txt
│   │   └── github_loader.py    # Load from GitHub REST API (issues + PRs)
│   │
│   └── router.py               # Pipeline coordinator
│
├── models/
│   ├── ltr_ranker.json         # Compiled XGBoost model
│   └── ltr_ranker.meta.json    # Training metadata
│
├── data/                       # Local datasets
│   ├── eclipse/
│   │   └── final dataset for work ecllipse.csv
│   └── bugzilla/
│       └── corpus (fixsev).txt
│
├── main.py                     # CLI Entry point
├── test_model.py               # Evaluation suite
├── train_ltr.py                # LTR Training script
├── requirements.txt
└── README.md
```

## Requirements

```text
numpy
sentence-transformers
xgboost
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

No API keys required. The GitHub API is used anonymously (rate limit: 60 requests/hour).

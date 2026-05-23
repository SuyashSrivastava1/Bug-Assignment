# Bug-to-Developer Assignment System

An AI-driven framework that automatically assigns software bug reports to the most suitable developer, using a hybrid pipeline of **NLP text analysis** (TF-IDF + KNN) and **Multi-Criteria Decision Making** (Weighted Sum Model).

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

You will be prompted to choose a mode:

| Mode | What you provide | What it does |
|---|---|---|
| **1 — GitHub URL** | A GitHub issue URL | Fetches the bug + the repo's full contributor history, then assigns the bug to the best developer from that project |
| **2 — Custom description** | Plain text | Uses the offline Eclipse dataset as training context and assigns the bug to the closest matching developer |

---

### Example — GitHub Mode

```
> 1
Paste a GitHub issue URL:
> https://github.com/facebook/react/issues/36469

[1/3] Fetching issue #36469 from facebook/react...
[2/3] Building developer pool from facebook/react history...
[3/3] Finding best developer...
--------------------------------------------------
  [NLP Signals Detected]
  Urgency    : 0.33  (crash)
  Complexity : 0.00
  Routine    : 0.00
  Severity   : 1.00  Effective Urgency: 0.72

  [Computed Weights for this Bug]
  Experience     0.108  ###
  Fix Time       0.361  ##########
  Success Rate   0.108  ###
  Workload       0.072  ##
  Domain Skill   0.351  ##########

  BEST MATCH : eps1lon
  Score      : 1.000
  Bugs Fixed : 5
  Avg Fix    : 11.5 hours

  Other Candidates:
    - rickhanlonii                   score=0.800  bugs_fixed=1
    - gnoff                          score=0.762  bugs_fixed=1
```

### Example — Custom Mode

```
> 2
> Memory leak in the worker thread pool causing high CPU usage

  [NLP Signals Detected]
  Urgency    : 0.00
  Complexity : 0.67  (memory, leak, threading)
  Routine    : 0.00
  Severity   : 0.50  Effective Urgency: 0.25

  [Computed Weights for this Bug]
  Experience     0.240  #######
  Fix Time       0.176  #####
  Success Rate   0.215  ######
  Workload       0.161  ####
  Domain Skill   0.209  ######

  BEST MATCH : Tod Creasey
  Score      : 0.693
```

---

## How the Algorithm Works

### 7-Step Pipeline

```
New Bug Report
      │
      ▼
1. TF-IDF Vectorisation  ──► Convert bug description text into a numeric feature vector
      │
      ▼
2. KNN Similarity Search ──► Find the top-K most similar bugs in the historical database
      │
      ▼
3. Candidate Extraction  ──► Identify the developers who fixed those similar bugs
      │
      ▼
4. Hard Filtering        ──► Remove developers who are on leave or over 95% workload
      │
      ▼
5. Attribute Matrix      ──► Build a matrix of 5 performance metrics for each candidate
      │
      ▼
6. NLP Dynamic Weighting ──► Analyse bug text to compute a UNIQUE weight vector per bug
      │
      ▼
7. WSM + Ranking         ──► Normalise, compute utility scores, return ranked list
```

### The 5 Developer Metrics

| Metric | Type | Source |
|---|---|---|
| **Experience** | Benefit ↑ | Total bugs historically fixed |
| **Fix Time** | Cost ↓ | Average hours to close a bug |
| **Success Rate** | Benefit ↑ | % of bugs not re-opened |
| **Workload** | Cost ↓ | Current open issue load (0–100%) |
| **Domain Skill** | Benefit ↑ | Number of distinct components worked on |

### Per-Bug Dynamic Weight Engine

Unlike traditional systems that pick from a fixed lookup table (e.g. "critical → fast fix"), this system reads the actual text of every bug report and computes a **unique weight vector** for each one.

Three NLP signals are extracted from the bug description:

| Signal | Example Keywords | Effect |
|---|---|---|
| **Urgency** | `crash`, `outage`, `production`, `security`, `500` | Increases weight on **Fix Time** and **Domain Skill** |
| **Complexity** | `memory leak`, `threading`, `deadlock`, `algorithm` | Increases weight on **Experience** and **Success Rate** |
| **Routine** | `typo`, `css`, `alignment`, `padding`, `cosmetic` | Increases weight on **Workload** (assign to least busy dev) |

**Severity** acts as a continuous amplifier (critical=1.0, normal=0.5, minor=0.0) that boosts the urgency signal — it does not select from a fixed table.

This means two bugs with the same severity label get different weights if their descriptions differ. A *"production database deadlock crash"* and a *"production crash on the login page"* are both `critical` but will correctly receive different weight distributions.

### Why GitHub + Eclipse Together?

When routing a GitHub bug, the system uses two data sources:

- **Eclipse dataset (10,000 bugs)** — as *NLP context only*. This gives the TF-IDF model a much richer vocabulary for understanding technical bug descriptions, improving similarity search accuracy.
- **GitHub repo history** — as the *candidate pool*. Only developers who actually work on that specific project can ever be assigned.

This means the AI is **smarter** (Eclipse NLP context) but always **practical** (only real project contributors are recommended).

---

## Evaluation Results

Evaluated on an 80/20 train/test split of the Eclipse dataset (9,800 train / 2,000 test bugs, 209 developers):

| Metric | Score |
|---|---|
| **Top-1 Accuracy** | 22.5% |
| **Top-3 Accuracy** | 51.5% |
| **Top-5 Accuracy** | 55.5% |
| **Mean Reciprocal Rank (MRR)** | 0.361 |

> **What these mean:** Top-3 accuracy of 51.5% means the correct developer appears in the AI's top 3 recommendations more than half the time. In practice, giving a project manager a ranked shortlist of 3 candidates is highly effective.

Run the evaluation yourself:

```bash
python test_model.py           # Full evaluation (~2,000 test bugs)
python test_model.py --quick   # Fast evaluation (200 test bugs)
python test_model.py --unit-only  # Unit + edge case tests only, no CSV needed
```

---

## Project Structure

```
Bug Classification/
│
├── src/                        # All source code
│   ├── models/
│   │   ├── bug.py              # Bug data class
│   │   ├── developer.py        # Developer data class (5 metrics)
│   │   └── assigner.py         # Core algorithm: TF-IDF, KNN, dynamic weights, WSM
│   │
│   ├── loaders/
│   │   ├── base.py             # Shared utilities: map_severity(), build_developers()
│   │   ├── csv_loader.py       # Load from Eclipse/Kaggle CSV dataset
│   │   └── github_loader.py    # Load from GitHub REST API (issues + PRs)
│   │
│   └── router.py               # Pipeline coordinator: ties all components together
│
├── documentation/              # Reference research papers
├── main.py                     # Entry point — run this
├── test_model.py               # Evaluation suite (unit, edge case, accuracy tests)
├── requirements.txt
└── README.md
```

## Requirements

```
numpy
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

No API keys required. The GitHub API is used anonymously (rate limit: 60 requests/hour). The Eclipse dataset CSV must be placed at `archive/final dataset for work ecllipse.csv` for CSV mode and accuracy evaluation to work.

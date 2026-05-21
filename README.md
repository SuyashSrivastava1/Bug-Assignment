# Bug-to-Developer Assignment System

An AI-driven framework that automatically assigns software bug reports to the most suitable developer, using a hybrid pipeline of NLP (TF-IDF + KNN) and Multi-Criteria Decision Making (MCDM).

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

You will be prompted to choose a mode:

| Mode | What you provide | What it does |
|---|---|---|
| **1 — GitHub URL** | A GitHub issue URL | Fetches the bug + the repo's full contributor history, then assigns the bug to the best developer from that project |
| **2 — Custom description** | Plain text | Uses the offline Eclipse dataset as training context and assigns the bug to the closest matching Eclipse developer |

### Example — GitHub Mode

```
> 1
Paste a GitHub issue URL:
> https://github.com/facebook/react/issues/36469

[1/3] Fetching issue #36469 from facebook/react...
[2/3] Building developer pool from facebook/react history...
[3/3] Finding best developer...
--------------------------------------------------
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
> NullPointerException in the Java package viewer UI

  BEST MATCH : Tod Creasey
  Score      : 0.693
  Bugs Fixed : 459
  Avg Fix    : 2053.7 hours
```

---

## How the Algorithm Works

### 7-Step Pipeline

```
New Bug Report
      │
      ▼
1. TF-IDF Vectorisation  ←── Convert bug description text into a numeric feature vector
      │
      ▼
2. KNN Similarity Search ←── Find the top-K most similar bugs in the historical database
      │
      ▼
3. Candidate Extraction  ←── Identify the developers who fixed those similar bugs
      │
      ▼
4. Hard Filtering        ←── Remove developers who are on leave or over 95% workload
      │
      ▼
5. Attribute Matrix      ←── Build a matrix of 5 performance metrics for each candidate
      │
      ▼
6. Dynamic Weighting     ←── Assign criteria weights based on bug severity
      │
      ▼
7. WSM + Ranking         ←── Normalise, compute utility scores, return ranked list
```

### The 5 Developer Metrics

| Metric | Type | Source |
|---|---|---|
| **Experience** | Benefit ↑ | Total bugs historically fixed |
| **Fix Time** | Cost ↓ | Average hours to close a bug |
| **Success Rate** | Benefit ↑ | % of bugs not re-opened |
| **Workload** | Cost ↓ | Current open issue load (0–100%) |
| **Domain Skill** | Benefit ↑ | Number of distinct components worked on |

### Dynamic Weights by Severity

| Severity | Experience | Fix Time | Success Rate | Workload | Domain Skill |
|---|---|---|---|---|---|
| **Critical** | 0.10 | **0.40** | 0.10 | 0.10 | **0.30** |
| **Normal** | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| **Minor** | 0.10 | 0.10 | 0.10 | **0.60** | 0.10 |

Critical bugs prioritise **speed** and **expertise**. Minor bugs prioritise assigning to the least busy developer.

### Why GitHub + Eclipse Together?

When routing a GitHub bug, the system uses:
- **Eclipse (10,000 bugs)** — as *NLP context only*. This gives the TF-IDF model a much richer vocabulary for understanding technical bug descriptions.
- **GitHub repo history** — as the *candidate pool*. Only developers who actually work on that specific project can be assigned.

This means the AI is smarter (Eclipse context) but always practical (only real contributors are recommended).

---

## Project Structure

```
Bug Classification/
│
├── src/                        # All source code
│   ├── models/
│   │   ├── bug.py              # Bug data class
│   │   ├── developer.py        # Developer data class (with all 5 metrics)
│   │   └── assigner.py         # Core KNN + WSM algorithm
│   │
│   ├── loaders/
│   │   ├── base.py             # Shared: map_severity(), build_developers()
│   │   ├── csv_loader.py       # Load from Eclipse/Kaggle CSV
│   │   └── github_loader.py    # Load from GitHub REST API (issues + PRs)
│   │
│   └── router.py               # Pipeline coordinator
│
├── archive/
│   └── final dataset for work ecllipse.csv
│
├── main.py                     # Entry point — run this
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

No API keys are required. The GitHub API is used anonymously (rate limit: 60 requests/hour).

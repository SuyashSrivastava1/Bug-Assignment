# Bug-to-Developer Assignment Framework

This project contains a Python implementation of an AI-driven, Multi-Criteria Decision Making (MCDM) framework designed to automate the assignment of software bugs to the optimal developers.

## Overview
The system relies on a hybrid pipeline that bridges semantic Natural Language Processing (NLP) with deterministic operational research models. It evaluates developers dynamically based on historical data, bug context, and team constraints to mathematically determine the best developer for a specific issue.

## Core Algorithm Steps
1. **Feature Extraction (NLP):** When a new bug is reported, its text is converted into a feature vector using TF-IDF Vectorization.
2. **Historical Similarity Search (KNN):** Cosine Similarity is used to find the top $K$ most similar historical bugs.
3. **Candidate Identification & Hard Filtering:** The developers who resolved the top $K$ similar bugs are identified. Developers on leave or over a 95% workload threshold are filtered out.
4. **Attribute Matrix Generation:** Candidates are evaluated on Experience (Benefit), Success Rate (Benefit), Domain Skill (Benefit), Fix Time (Cost), and Workload (Cost).
5. **Dynamic Contextual AI Weighting:** Weights are dynamically assigned based on bug severity (e.g., Critical bugs prioritize speed and skill, while Minor bugs prioritize low workload).
6. **Normalization and WSM Calculation:** Metrics are normalized to a `[0, 1]` scale using Min-Max scaling. The Weighted Sum Model (WSM) computes the final Utility Score.
7. **Ranking and Assignment:** Developers are ranked descending by Utility Score to find the mathematical optimum assignment.

---

## The Code Structure

There are three ways to run and test this algorithm depending on the data source you want to use.

### 1. The Core Algorithm (Mock Data)
**File:** `bug_assigner.py`
This is the core logic file. When run directly, it executes a simple test using hard-coded mock data (5 developers and 5 historical bugs). It is useful for understanding the exact flow of the algorithm without needing external data.
```bash
python bug_assigner.py
```

### 2. The Kaggle Dataset Pipeline (Real-World CSV Data)
**File:** `csv_pipeline.py`
This script is specifically configured to load the **Eclipse Bug Triaging Dataset** downloaded from Kaggle (`archive/final dataset for work ecllipse.csv`). 

**What it does:**
* It parses the Kaggle CSV, extracting `Bug ID`, `Assignee Real Name`, and computes fix times using the `Opened` and `Changed` timestamps.
* It loads all 10,000 bugs from the Eclipse dataset.
* It trains the AI's "Historical Knowledge" on the first 8,000 bugs.
* It then tests the algorithm's assignment predictions on the remaining 2,000 test bugs.

**How to run it:**
```bash
python csv_pipeline.py
```

### 3. The Live GitHub Pipeline (Real-Time API Data)
**File:** `github_pipeline.py`
This script queries the actual GitHub REST API to pull the 100 most recently closed bugs from the massive **Microsoft VS Code** repository (`microsoft/vscode`).

**What it does:**
* Hits the live GitHub API (no CSV required).
* Filters out pull requests and extracts labels, titles, descriptions, and assignees.
* Trains the AI on the first 80 bugs and tests the assignment on the remaining 20 bugs.
* Prints out the AI's optimal developer choices and compares them to the developer who *actually* fixed the bug in reality.

**How to run it:**
```bash
python github_pipeline.py
```

### 4. The Unified Pipeline (Mixed Multi-Dataset Training)
**File:** `unified_pipeline.py`
This script demonstrates the ultimate scale of the Bug Assigner AI by training it simultaneously on **multiple massive, completely distinct datasets**. 

**What it does:**
* Loads all 10,000 bugs from the offline Eclipse CSV.
* Concurrently fetches live recent bugs from the VS Code GitHub API.
* Merges everything into a massive pool of 10,000+ historical bugs and hundreds of unique developers.
* Accepts a brand-new, completely custom bug report and routes it to the absolute best developer out of the entire global pool.

**How to run it:**
```bash
python unified_pipeline.py
```

### 5. The Interactive Command Line (No Dummy Data)
**File:** `interactive_cli.py`
This script gives you an interactive prompt where you can type in your own custom bug descriptions and instantly get assigned a real developer.

**What it does:**
* It completely removes all randomized/dummy metrics (setting missing data to neutral baselines).
* It trains the AI on all 10,000 real Eclipse bugs.
* It pauses and asks you to type in a bug description.
* It evaluates your exact text and tells you exactly which real Eclipse developer is best suited to fix it based strictly on their actual historical track record.

**How to run it:**
```bash
python interactive_cli.py
```

## Requirements
To run any of the scripts, ensure you have the required dependencies installed:
```bash
pip install numpy scikit-learn
```

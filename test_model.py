"""
test_model.py — Evaluation suite for the Bug-to-Developer Assignment System.

Runs four types of tests:

  1. Unit Tests         — Verify Bug, Developer, and BugAssigner work in isolation.
  2. Loader Tests       — Verify the CSV loader parses the Eclipse dataset correctly.
  3. Accuracy Tests     — Train/test split on Eclipse data; measure Top-1/3/5 accuracy,
                          Mean Reciprocal Rank (MRR), and per-severity breakdown.
  4. Edge Case Tests    — Empty descriptions, unknown severity, all developers on leave, etc.

Accuracy Metrics Explained
--------------------------
  Top-1 Accuracy : The correct developer is the #1 recommendation.
  Top-3 Accuracy : The correct developer appears anywhere in the top-3 recommendations.
  Top-5 Accuracy : The correct developer appears anywhere in the top-5 recommendations.
  MRR            : Mean Reciprocal Rank — 1/rank of the first correct answer, averaged
                   across all test bugs. A perfect score is 1.0.

These are the standard metrics used in academic bug triaging research.

Usage
-----
    python test_model.py               # Run all tests
    python test_model.py --quick       # Run with a smaller test set (faster)
    python test_model.py --unit-only   # Skip accuracy tests (no CSV needed)
"""

import sys
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Evaluate the Bug Assignment model.")
parser.add_argument("--quick",     action="store_true", help="Use 200 test bugs instead of all")
parser.add_argument("--unit-only", action="store_true", help="Run only unit + edge case tests")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

PASS  = "[PASS]"
FAIL  = "[FAIL]"
SKIP  = "[SKIP]"
INFO  = "[INFO]"

_results = {"passed": 0, "failed": 0, "skipped": 0}


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test(name: str, condition: bool, detail: str = ""):
    if condition:
        _results["passed"] += 1
        print(f"  {PASS}  {name}")
    else:
        _results["failed"] += 1
        msg = f"  {FAIL}  {name}"
        if detail:
            msg += f"\n         Detail: {detail}"
        print(msg)


def skip(name: str, reason: str = ""):
    _results["skipped"] += 1
    msg = f"  {SKIP}  {name}"
    if reason:
        msg += f"  ({reason})"
    print(msg)


def info(msg: str):
    print(f"  {INFO}  {msg}")


# ===========================================================================
# SECTION 1 — Unit Tests
# ===========================================================================

section("1. Unit Tests — Models")

from src.models.bug import Bug
from src.models.developer import Developer
from src.models.assigner import BugAssigner, compute_dynamic_weights

# Bug
b = Bug(id=1, description="App crashes on startup", severity="CRITICAL", module="core")
test("Bug: id stored correctly",          b.id == 1)
test("Bug: description stored correctly", b.description == "App crashes on startup")
test("Bug: severity lowercased",          b.severity == "critical")
test("Bug: module stored correctly",      b.module == "core")
test("Bug: vector is None before fit",    b.vector is None)

# Developer
d = Developer(id=7, name="Alice", experience=50, fix_time=3.5,
              success_rate=95, workload=40, domain_skill=80)
test("Developer: all fields stored",   d.name == "Alice" and d.experience == 50)
test("Developer: on_leave defaults False", d.on_leave is False)

# BugAssigner — untrained
assigner = BugAssigner()
try:
    assigner.assign(b)
    test("BugAssigner: raises RuntimeError before fit", False, "No exception was raised")
except RuntimeError:
    test("BugAssigner: raises RuntimeError before fit", True)

# BugAssigner — minimal training
bugs_train = [
    Bug(101, "crash in login module",        "critical", "auth"),
    Bug(102, "CSS misalignment on homepage", "minor",    "ui"),
    Bug(103, "memory leak in worker thread", "critical", "backend"),
]
devs = [
    Developer(1, "Alice", experience=10, fix_time=2.0,  success_rate=100, workload=0, domain_skill=70),
    Developer(2, "Bob",   experience=5,  fix_time=5.0,  success_rate=100, workload=0, domain_skill=60),
    Developer(3, "Carol", experience=3,  fix_time=8.0,  success_rate=100, workload=0, domain_skill=50),
]
resolutions = {101: 1, 102: 2, 103: 1}

assigner.fit(bugs_train, devs, resolutions)
test("BugAssigner: fit() completes without error", assigner._historical_dense is not None)

new_bug = Bug(200, "crash when logging in as admin user", "critical", "auth")
best, rankings, weights, signals = assigner.assign(new_bug, k=3)
test("BugAssigner: assign() returns a developer",    best is not None)
test("BugAssigner: rankings is a non-empty list",    len(rankings) > 0)
test("BugAssigner: weights are returned",            weights is not None and len(weights) == 6)
test("BugAssigner: weights sum to 1.0",              abs(float(weights.sum()) - 1.0) < 1e-6)
test("BugAssigner: signals dict is returned",        isinstance(signals, dict) and "urgency_score" in signals)
test("BugAssigner: best dev is top of rankings",     rankings[0][0].name == best.name)
test("BugAssigner: scores are descending",
     all(rankings[i][1] >= rankings[i+1][1] for i in range(len(rankings)-1)))

# Dynamic weight engine
section("1b. Unit Tests — Dynamic Weight Engine")

# Every bug gets a unique weight vector
bug_crash = Bug(1, "production outage crash critical server down", "critical", "backend")
bug_css   = Bug(2, "typo alignment css padding margin cosmetic",   "minor",    "ui")
bug_leak  = Bug(3, "memory leak threading concurrent deadlock",    "normal",   "backend")

w_crash, s_crash = compute_dynamic_weights(bug_crash)
w_css,   s_css   = compute_dynamic_weights(bug_css)
w_leak,  s_leak  = compute_dynamic_weights(bug_leak)

test("DynWeights: crash bug sums to 1.0",  abs(float(w_crash.sum()) - 1.0) < 1e-6)
test("DynWeights: css bug sums to 1.0",    abs(float(w_css.sum())   - 1.0) < 1e-6)
test("DynWeights: leak bug sums to 1.0",   abs(float(w_leak.sum())  - 1.0) < 1e-6)
test("DynWeights: no negatives in weights", all(v >= 0 for w in [w_crash, w_css, w_leak] for v in w))

# Verify signal direction — crash bug should weight FixTime (index 1) heavily
test("DynWeights: crash bug weights FixTime highest",
     w_crash[1] == max(w_crash), f"Weights: {[round(v,3) for v in w_crash]}")

# CSS/cosmetic bug should weight Workload (index 3) heavily
test("DynWeights: css bug weights Workload highest",
     w_css[3] == max(w_css), f"Weights: {[round(v,3) for v in w_css]}")

# Complex bug should weight Experience (0) or SuccessRate (2) heavily
test("DynWeights: complex bug Experience > minor threshold",
     w_leak[0] > 0.15, f"Experience weight: {w_leak[0]:.3f}")

# Each bug should get different weights
test("DynWeights: crash != css (uniqueness)",  list(w_crash) != list(w_css))
test("DynWeights: crash != leak (uniqueness)", list(w_crash) != list(w_leak))
test("DynWeights: css != leak (uniqueness)",   list(w_css)   != list(w_leak))

# Signals dict has the required keys
required_keys = {"urgency_score", "complexity_score", "routine_score",
                 "severity_score", "effective_urgency",
                 "urgency_keywords", "complexity_keywords", "routine_keywords"}
test("DynWeights: signals dict has all keys", required_keys.issubset(s_crash.keys()))

# Urgency keywords should be detected for the crash bug
test("DynWeights: crash signals contain urgency keywords",
     len(s_crash["urgency_keywords"]) > 0,
     f"Found: {s_crash['urgency_keywords']}")

# Routine keywords should be detected for the css bug
test("DynWeights: css signals contain routine keywords",
     len(s_css["routine_keywords"]) > 0,
     f"Found: {s_css['routine_keywords']}")

# Complexity keywords should be detected for the leak bug
test("DynWeights: leak signals contain complexity keywords",
     len(s_leak["complexity_keywords"]) > 0,
     f"Found: {s_leak['complexity_keywords']}")

# Score bounds
test("DynWeights: urgency_score in [0, 1]",    0.0 <= s_crash["urgency_score"]    <= 1.0)
test("DynWeights: complexity_score in [0, 1]", 0.0 <= s_leak["complexity_score"]  <= 1.0)
test("DynWeights: routine_score in [0, 1]",    0.0 <= s_css["routine_score"]      <= 1.0)
test("DynWeights: effective_urgency in [0, 1]",0.0 <= s_crash["effective_urgency"] <= 1.0)

# ===========================================================================
# SECTION 2 — Edge Case Tests
# ===========================================================================

section("2. Edge Case Tests")

# Empty description
edge_bug = Bug(999, "", "normal", "general")
best_e, rankings_e, weights_e, signals_e = assigner.assign(edge_bug, k=3)
test("Edge: empty description doesn't crash", True)  # If we get here, no exception was raised

# Unknown severity falls back to normal (severity_score=0.5)
edge_bug2 = Bug(998, "some bug", "UNKNOWN_SEVERITY", "ui")
_, _, w2, s2 = assigner.assign(edge_bug2, k=3)
test("Edge: unknown severity uses fallback weights", w2 is not None and abs(float(w2.sum()) - 1.0) < 1e-6)
test("Edge: unknown severity has severity_score 0.5", abs(s2["severity_score"] - 0.5) < 1e-6)

# All developers on leave
devs_leave = [
    Developer(1, "OnLeave", experience=10, fix_time=2.0, success_rate=100, workload=0, domain_skill=70, on_leave=True)
]
assigner_leave = BugAssigner()
assigner_leave.fit(bugs_train, devs_leave, resolutions)
best_l, _, _, _ = assigner_leave.assign(new_bug, k=3)
test("Edge: all on-leave -> returns None", best_l is None)

# All developers overloaded (workload > 95)
devs_overloaded = [
    Developer(1, "Overloaded", experience=10, fix_time=2.0, success_rate=100, workload=99, domain_skill=70)
]
assigner_over = BugAssigner()
assigner_over.fit(bugs_train, devs_overloaded, resolutions)
best_o, _, _, _ = assigner_over.assign(new_bug, k=3)
test("Edge: all overloaded -> returns None", best_o is None)

# Single developer (no comparison possible, everyone gets 1.0)
single_dev = [Developer(1, "Solo", experience=10, fix_time=2.0, success_rate=100, workload=0, domain_skill=70)]
assigner_single = BugAssigner()
assigner_single.fit(bugs_train, single_dev, {101: 1, 102: 1, 103: 1})
best_s, rankings_s, _, _ = assigner_single.assign(new_bug, k=3)
test("Edge: single developer is returned as best", best_s is not None and best_s.name == "Solo")


# ===========================================================================
# SECTION 3 — Loader Tests
# ===========================================================================

section("3a. Loader Tests — Eclipse CSV")

CSV_PATH = Path("data") / "eclipse" / "final dataset for work ecllipse.csv"

if not CSV_PATH.exists():
    skip("CSV load: file exists",         f"Not found at {CSV_PATH}")
    skip("CSV load: returns > 0 bugs",    "Skipped — no CSV")
    skip("CSV load: returns > 0 devs",    "Skipped — no CSV")
    skip("CSV load: all bugs have ids",   "Skipped — no CSV")
    skip("CSV load: no empty summaries",  "Skipped — no CSV")
    csv_bugs, csv_dev_stats = None, None
else:
    from src.loaders.csv_loader import load as csv_load
    t0 = time.time()
    csv_bugs, csv_dev_stats = csv_load(CSV_PATH)
    elapsed = time.time() - t0

    test("CSV load: file parses without error",   True)
    test("CSV load: returns > 0 bugs",            len(csv_bugs) > 0,   f"Got {len(csv_bugs)}")
    test("CSV load: returns > 0 devs",            len(csv_dev_stats) > 0)
    test("CSV load: all bugs have non-empty id",  all(b.id for b in csv_bugs))
    test("CSV load: no empty descriptions",       all(b.description.strip() for b in csv_bugs))
    test("CSV load: all severities are valid",
         all(b.severity in ("critical", "normal", "minor") for b in csv_bugs))
    info(f"Loaded {len(csv_bugs):,} bugs and {len(csv_dev_stats):,} developers in {elapsed:.2f}s")


# ===========================================================================
# SECTION 3b — Loader Tests: Bugzilla corpus
# ===========================================================================

section("3b. Loader Tests — Bugzilla Corpus")

from src.loaders.bugzilla_loader import load as bugzilla_load, DEFAULT_BUGZILLA_PATH

# Always test: missing file must never crash the pipeline
bzl_missing_bugs, bzl_missing_devs = bugzilla_load(Path("nonexistent_path_xyz.txt"))
test("Bugzilla: missing file returns empty list (no crash)",
     isinstance(bzl_missing_bugs, list) and len(bzl_missing_bugs) == 0)
test("Bugzilla: missing file returns empty dev_stats",
     isinstance(bzl_missing_devs, dict) and len(bzl_missing_devs) == 0)

if not DEFAULT_BUGZILLA_PATH.exists():
    skip("Bugzilla load: file present",           f"Not found at {DEFAULT_BUGZILLA_PATH}")
    skip("Bugzilla load: returns > 0 bugs",       "Skipped — corpus not downloaded")
    skip("Bugzilla load: dev_stats always empty", "Skipped — corpus not downloaded")
    skip("Bugzilla load: all descriptions non-empty", "Skipped — corpus not downloaded")
    skip("Bugzilla load: all severities valid",   "Skipped — corpus not downloaded")
    skip("Bugzilla load: bug ids are unique",      "Skipped — corpus not downloaded")
    bzl_bugs = None
else:
    t0 = time.time()
    bzl_bugs, bzl_devs = bugzilla_load(DEFAULT_BUGZILLA_PATH)
    elapsed_bzl = time.time() - t0

    test("Bugzilla load: file parses without error", True)
    test("Bugzilla load: returns > 0 bugs",
         len(bzl_bugs) > 0, f"Got {len(bzl_bugs)}")
    test("Bugzilla load: dev_stats is always empty (context-only)",
         len(bzl_devs) == 0)
    test("Bugzilla load: all descriptions non-empty",
         all(b.description.strip() for b in bzl_bugs))
    test("Bugzilla load: all severities valid",
         all(b.severity in ("critical", "normal", "minor") for b in bzl_bugs))
    all_ids = [b.id for b in bzl_bugs]
    test("Bugzilla load: bug ids are unique",
         len(all_ids) == len(set(all_ids)))
    info(f"Loaded {len(bzl_bugs):,} Bugzilla bug descriptions in {elapsed_bzl:.2f}s")


# ===========================================================================
# SECTION 4 — Accuracy Evaluation (Train / Test Split)
# ===========================================================================

section("4. Accuracy Evaluation — Train / Test Split on Eclipse Dataset")

if args.unit_only:
    skip("Accuracy evaluation", "--unit-only flag set")
elif csv_bugs is None:
    skip("Accuracy evaluation", "CSV dataset not available")
else:
    from src.loaders.base import build_developers

    # -----------------------------------------------------------------------
    # Build full developer pool (all bugs)
    # We need the full pool so test-set developers are known to the assigner.
    # -----------------------------------------------------------------------
    all_bugs = csv_bugs
    all_devs, all_resolutions = build_developers(csv_dev_stats)

    # Only evaluate bugs that have a known resolution (i.e., appear in resolutions map)
    # and whose assignee resolves to a real Developer object
    dev_id_by_name = {d.name: d.id for d in all_devs}
    resolvable = [
        b for b in all_bugs
        if b.id in all_resolutions
    ]

    # Shuffle deterministically then split 80/20
    import random
    rng = random.Random(42)
    rng.shuffle(resolvable)

    n_test = 200 if args.quick else max(200, len(resolvable) // 5)
    n_test = min(n_test, len(resolvable))

    test_bugs   = resolvable[:n_test]
    train_bugs  = resolvable[n_test:]

    info(f"Total resolvable bugs : {len(resolvable):,}")
    info(f"Training set          : {len(train_bugs):,} bugs")
    info(f"Test set              : {n_test:,} bugs")
    info(f"Developer pool        : {len(all_devs):,} unique developers")

    # Train the assigner on training bugs only, but use all developers as the pool
    assigner_eval = BugAssigner()
    assigner_eval.fit(train_bugs, all_devs, all_resolutions)

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    top1 = top3 = top5 = mrr_sum = 0
    no_assignment = 0
    severity_counts   = {"critical": 0, "normal": 0, "minor": 0}
    severity_top1     = {"critical": 0, "normal": 0, "minor": 0}

    print()
    print("  Running evaluation ...")
    t_start = time.time()

    for bug in test_bugs:
        actual_dev_id = all_resolutions.get(bug.id)
        if actual_dev_id is None:
            no_assignment += 1
            continue

        best, rankings, _, _ = assigner_eval.assign(bug, k=20)
        if best is None:
            no_assignment += 1
            continue

        ranked_ids = [dev.id for dev, _ in rankings]
        actual_pos = None
        for rank, dev_id in enumerate(ranked_ids, start=1):
            if dev_id == actual_dev_id:
                actual_pos = rank
                break

        sev = bug.severity
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        if actual_pos is not None:
            mrr_sum += 1.0 / actual_pos
            if actual_pos == 1:
                top1 += 1
                severity_top1[sev] = severity_top1.get(sev, 0) + 1
            if actual_pos <= 3:
                top3 += 1
            if actual_pos <= 5:
                top5 += 1

    t_eval = time.time() - t_start
    evaluated = n_test - no_assignment

    info(f"Evaluation time : {t_eval:.2f}s  ({t_eval/max(evaluated,1)*1000:.1f} ms/bug)")
    info(f"Evaluated       : {evaluated:,}  (skipped {no_assignment} with no resolution)")

    top1_pct = (top1 / evaluated * 100) if evaluated else 0
    top3_pct = (top3 / evaluated * 100) if evaluated else 0
    top5_pct = (top5 / evaluated * 100) if evaluated else 0
    mrr      = (mrr_sum / evaluated)    if evaluated else 0
    no_assignment_pct = (no_assignment / n_test * 100) if n_test else 0
    ms_per_bug = (t_eval / max(evaluated, 1)) * 1000

    print()
    print(f"  {'Metric':<28}  {'Value':>10}  {'Pass Threshold':>16}")
    print(f"  {'-'*58}")
    print(f"  {'Top-1 Accuracy':<28}  {top1_pct:>9.2f}%  {'> 15%':>16}")
    print(f"  {'Top-3 Accuracy':<28}  {top3_pct:>9.2f}%  {'> 30%':>16}")
    print(f"  {'Top-5 Accuracy':<28}  {top5_pct:>9.2f}%  {'> 40%':>16}")
    print(f"  {'Mean Reciprocal Rank (MRR)':<28}  {mrr:>10.4f}  {'> 0.15':>16}")
    print(f"  {'No-assignment rate':<28}  {no_assignment_pct:>9.2f}%  {'< 5%':>16}")
    print(f"  {'Eval latency (ms/bug)':<28}  {ms_per_bug:>10.1f}  {'< 750':>16}")
    print()

    # Per-severity breakdown
    severity_top1_pct = {}
    print(f"  Per-Severity Top-1 Accuracy:")
    for sev in ("critical", "normal", "minor"):
        cnt = severity_counts.get(sev, 0)
        if cnt > 0:
            pct = severity_top1.get(sev, 0) / cnt * 100
            severity_top1_pct[sev] = pct
            print(f"    {sev:<12}: {pct:.2f}%  ({cnt} bugs)")

    # Record accuracy results as pass/fail
    print()
    test("Top-1 Accuracy > 15%",  top1_pct > 15.0,  f"Got {top1_pct:.2f}%")
    test("Top-3 Accuracy > 30%",  top3_pct > 30.0,  f"Got {top3_pct:.2f}%")
    test("Top-5 Accuracy > 40%",  top5_pct > 40.0,  f"Got {top5_pct:.2f}%")
    test("MRR > 0.15",            mrr > 0.15,        f"Got {mrr:.4f}")
    test("No-assignment rate < 5%", no_assignment_pct < 5.0, f"Got {no_assignment_pct:.2f}%")
    test("Evaluated coverage > 95%", evaluated / max(n_test, 1) > 0.95, f"Got {evaluated}/{n_test}")
    test("Top-k monotonicity (Top-1 <= Top-3 <= Top-5)", top1_pct <= top3_pct <= top5_pct,
         f"Got Top-1={top1_pct:.2f}, Top-3={top3_pct:.2f}, Top-5={top5_pct:.2f}")
    test("Top-3 lift over Top-1 > 15pp", (top3_pct - top1_pct) > 15.0,
         f"Got +{(top3_pct - top1_pct):.2f}pp")
    test("Top-5 lift over Top-3 > 4pp", (top5_pct - top3_pct) > 4.0,
         f"Got +{(top5_pct - top3_pct):.2f}pp")
    test("Evaluation latency < 750 ms/bug", ms_per_bug < 750.0, f"Got {ms_per_bug:.1f} ms/bug")

    severity_thresholds = {
        "critical": 10.0,
        "normal": 20.0,
        "minor": 20.0,
    }
    for sev, threshold in severity_thresholds.items():
        cnt = severity_counts.get(sev, 0)
        if cnt == 0:
            skip(f"{sev.title()} Top-1 > {threshold:.0f}%", "No samples in test split")
            continue
        pct = severity_top1_pct[sev]
        test(f"{sev.title()} Top-1 > {threshold:.0f}%", pct > threshold, f"Got {pct:.2f}%")


# ===========================================================================
# Summary
# ===========================================================================

section("Test Summary")
total = _results["passed"] + _results["failed"] + _results["skipped"]
print(f"  Passed  : {_results['passed']}")
print(f"  Failed  : {_results['failed']}")
print(f"  Skipped : {_results['skipped']}")
print(f"  Total   : {total}")
print()

if _results["failed"] > 0:
    print("  RESULT: SOME TESTS FAILED")
    sys.exit(1)
else:
    print("  RESULT: ALL TESTS PASSED")
    sys.exit(0)

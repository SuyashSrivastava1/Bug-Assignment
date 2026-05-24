"""
compare_models.py — Side-by-side comparison of WSM vs LTR.
"""
from pathlib import Path
import time
from src.loaders.csv_loader import load as csv_load
from src.loaders.base import build_developers
from src.models.assigner import BugAssigner

def evaluate(assigner, test_bugs, all_resolutions, use_ltr=False):
    assigner._use_ltr = use_ltr
    top1 = top3 = top5 = mrr_sum = 0
    no_assignment = 0

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
        actual_pos = None
        for rank, dev_id in enumerate(ranked_ids, start=1):
            if dev_id == actual_dev_id:
                actual_pos = rank
                break

        if actual_pos is not None:
            mrr_sum += 1.0 / actual_pos
            if actual_pos == 1:
                top1 += 1
            if actual_pos <= 3:
                top3 += 1
            if actual_pos <= 5:
                top5 += 1

    t_eval = time.time() - t_start
    evaluated = len(test_bugs) - no_assignment

    top1_pct = (top1 / evaluated * 100) if evaluated else 0
    top3_pct = (top3 / evaluated * 100) if evaluated else 0
    top5_pct = (top5 / evaluated * 100) if evaluated else 0
    mrr = (mrr_sum / evaluated) if evaluated else 0
    latency = (t_eval / max(evaluated, 1)) * 1000

    return top1_pct, top3_pct, top5_pct, mrr, latency

if __name__ == "__main__":
    CSV_PATH = Path("data") / "eclipse" / "final dataset for work ecllipse.csv"
    bugs, dev_stats = csv_load(CSV_PATH)
    devs, resolutions = build_developers(dev_stats)

    resolvable = [b for b in bugs if b.id in resolutions]

    import random
    rng = random.Random(42)
    rng.shuffle(resolvable)

    n_test = 200
    test_bugs = resolvable[:n_test]
    train_bugs = resolvable[n_test:]

    assigner = BugAssigner()
    assigner.fit(train_bugs, devs, resolutions)
    
    # Load LTR model
    assigner.load_ltr_model("models/ltr_ranker.json")

    print("Evaluating WSM...")
    wsm_metrics = evaluate(assigner, test_bugs, resolutions, use_ltr=False)
    
    print("Evaluating LTR...")
    ltr_metrics = evaluate(assigner, test_bugs, resolutions, use_ltr=True)

    print("\n   ---------------------------------")
    print("   | Metric  |  WSM   |  LTR   |")
    print("   ---------------------------------")
    print(f"   | Top-1   | {wsm_metrics[0]:.2f}% | {ltr_metrics[0]:.2f}% |")
    print(f"   | Top-3   | {wsm_metrics[1]:.2f}% | {ltr_metrics[1]:.2f}% |")
    print(f"   | Top-5   | {wsm_metrics[2]:.2f}% | {ltr_metrics[2]:.2f}% |")
    print(f"   | MRR     | {wsm_metrics[3]:.4f} | {ltr_metrics[3]:.4f} |")
    print(f"   | Latency | {wsm_metrics[4]:.1f}ms | {ltr_metrics[4]:.1f}ms |")
    print("   ---------------------------------")

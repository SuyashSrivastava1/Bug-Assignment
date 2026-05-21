"""
src/loaders/base.py

Shared helpers used by every data loader:
  - map_severity()      : Normalises raw severity strings to 'critical'/'normal'/'minor'.
  - build_developers()  : Converts a dev_stats dict into Developer objects + a resolutions map.

Every loader returns data in the same format so the router can combine them freely.
"""

from src.models.developer import Developer


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

_CRITICAL_LABELS = {"blocker", "critical", "crash", "major", "bug"}
_MINOR_LABELS    = {"trivial", "minor", "enhancement", "feature-request", "feature request"}


def map_severity(raw: str) -> str:
    """
    Normalise a raw severity/label string to one of: 'critical', 'normal', 'minor'.

    Parameters
    ----------
    raw : Any severity string from Bugzilla, GitHub labels, CSV columns, etc.

    Returns
    -------
    str — 'critical', 'minor', or 'normal' (default).
    """
    cleaned = (raw or "").lower().strip()
    if cleaned in _CRITICAL_LABELS:
        return "critical"
    if cleaned in _MINOR_LABELS:
        return "minor"
    return "normal"


# ---------------------------------------------------------------------------
# Developer builder
# ---------------------------------------------------------------------------

def build_developers(dev_stats: dict) -> tuple[list[Developer], dict]:
    """
    Convert a dev_stats dictionary into a list of Developer objects and a resolutions map.

    dev_stats format (produced by every loader):
    {
        "username_or_name": {
            "name":             str,
            "bugs_fixed":       int,
            "total_fix_time":   float,   # hours
            "components":       set[str],
            "resolved_bug_ids": list[int | str],
        },
        ...
    }

    All metrics are calculated from real historical data.
    Missing metrics (success_rate, workload) are set to neutral baselines:
      - success_rate = 100  (assume fully successful; penalise only when re-open data exists)
      - workload     = 0    (assume available; can be overridden when live data exists)

    Returns
    -------
    developers  : list[Developer]
    resolutions : dict mapping bug_id -> developer_id
    """
    developers: list[Developer] = []
    resolutions: dict = {}
    dev_id = 1

    for stats in dev_stats.values():
        n = stats["bugs_fixed"]
        if n == 0:
            continue

        experience   = n
        avg_fix_time = stats["total_fix_time"] / n
        domain_skill = min(100.0, 50.0 + len(stats["components"]) * 10.0)
        success_rate = 100.0   # neutral baseline
        workload     = 0.0     # neutral baseline

        dev = Developer(
            id=dev_id,
            name=stats["name"],
            experience=experience,
            fix_time=avg_fix_time,
            success_rate=success_rate,
            workload=workload,
            domain_skill=domain_skill,
        )
        developers.append(dev)

        for bug_id in stats.get("resolved_bug_ids", []):
            resolutions[bug_id] = dev_id

        dev_id += 1

    return developers, resolutions

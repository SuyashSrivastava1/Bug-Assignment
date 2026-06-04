"""
src/loaders/base.py

Shared utilities used by every data loader:
  - map_severity()     : Normalises raw severity strings → 'critical' / 'normal' / 'minor'.
  - build_developers() : Converts a dev_stats dict into Developer objects + a resolutions map.

All loaders return data in the same format so the Router can freely combine them.
"""

from __future__ import annotations

import logging
from src.models.developer import Developer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

_CRITICAL_LABELS: frozenset[str] = frozenset(
    {"blocker", "critical", "crash", "major", "bug"}
)
_MINOR_LABELS: frozenset[str] = frozenset(
    {"trivial", "minor", "enhancement", "feature-request", "feature request"}
)


def map_severity(raw: str) -> str:
    """
    Normalise a raw severity/label string to one of: 'critical', 'normal', 'minor'.

    Parameters
    ----------
    raw : Any severity string from Bugzilla, GitHub labels, CSV columns, etc.

    Returns
    -------
    'critical', 'minor', or 'normal' (default).
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

    Missing metrics use neutral baselines:
      - success_rate = 100  (penalise only when re-open data exists)
      - workload     = 0    (assume available; override when live data exists)

    Returns
    -------
    developers  : list[Developer]
    resolutions : dict mapping bug_id -> developer_id
    """
    developers: list[Developer] = []
    resolutions: dict = {}
    dev_id = 1

    for stats in dev_stats.values():
        n: int = stats["bugs_fixed"]
        if n == 0:
            continue

        avg_fix_time: float = stats["total_fix_time"] / n
        # Domain skill: 50 base + 10 per unique component worked on, capped at 100
        domain_skill: float = min(100.0, 50.0 + len(stats["components"]) * 10.0)

        dev = Developer(
            id=dev_id,
            name=stats["name"],
            experience=n,
            fix_time=avg_fix_time,
            success_rate=100.0,  # neutral baseline
            workload=0.0,        # neutral baseline
            domain_skill=domain_skill,
        )
        developers.append(dev)

        for bug_id in stats.get("resolved_bug_ids", []):
            resolutions[bug_id] = dev_id

        dev_id += 1

    logger.debug("Built %d developers, %d resolution mappings.", len(developers), len(resolutions))
    return developers, resolutions

"""
src/loaders/csv_loader.py

Loads historical bug data from a local CSV file (Eclipse Bug Triaging dataset).

Expected CSV columns:
    Bug ID, Summary, Severity, Component, Assignee Real Name, Opened, Changed

Returns
-------
bugs      : list[Bug]  — all valid assigned bugs from the file.
dev_stats : dict       — aggregated developer metrics (input to base.build_developers).
"""

from __future__ import annotations

import csv
import datetime
import logging
from pathlib import Path

from src.models.bug import Bug
from src.loaders.base import map_severity

logger = logging.getLogger(__name__)

# Default path relative to the project root
DEFAULT_CSV_PATH = Path("data") / "eclipse" / "final dataset for work ecllipse.csv"


def load(filepath: str | Path = DEFAULT_CSV_PATH) -> tuple[list[Bug], dict]:
    """
    Parse a CSV bug dataset and return bugs + dev_stats.

    Parameters
    ----------
    filepath : Path to the CSV file.

    Returns
    -------
    (bugs, dev_stats)

    Raises
    ------
    FileNotFoundError if the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV dataset not found at: {filepath}")

    bugs: list[Bug] = []
    dev_stats: dict = {}

    with open(filepath, mode="r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bug_id   = row.get("Bug ID", "").strip()
            summary  = row.get("Summary", "").strip()
            assignee = (row.get("Assignee Real Name") or row.get("Assignee", "")).strip()

            if not bug_id or not summary or not assignee:
                continue

            severity  = map_severity(row.get("Severity", "normal"))
            component = (row.get("Component") or "general").strip()
            created_ts, fix_time_hours = _parse_timestamps(
                row.get("Opened", ""), row.get("Changed", "")
            )

            bugs.append(Bug(
                id=bug_id, 
                description=summary, 
                severity=severity, 
                module=component,
                created_at=created_ts
            ))

            if assignee not in dev_stats:
                dev_stats[assignee] = {
                    "name": assignee,
                    "bugs_fixed": 0,
                    "total_fix_time": 0.0,
                    "components": set(),
                    "resolved_bug_ids": [],
                }

            dev_stats[assignee]["bugs_fixed"]      += 1
            dev_stats[assignee]["total_fix_time"]  += fix_time_hours
            dev_stats[assignee]["components"].add(component)
            dev_stats[assignee]["resolved_bug_ids"].append(bug_id)

    logger.info(
        "[CSV Loader] Loaded %d bugs from %s (%d unique developers).",
        len(bugs), filepath.name, len(dev_stats),
    )
    print(
        f"[CSV Loader] Loaded {len(bugs):,} bugs from {filepath.name} "
        f"({len(dev_stats):,} unique developers)"
    )
    return bugs, dev_stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_timestamps(
    opened: str,
    changed: str,
    fallback_hours: float = 24.0,
) -> tuple[float, float]:
    """Return the (created_timestamp, fix_time_hours)."""
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t0 = datetime.datetime.strptime(opened.strip(), fmt)
        t1 = datetime.datetime.strptime(changed.strip(), fmt)
        fix_time = max(0.1, (t1 - t0).total_seconds() / 3600.0)
        return (t0.timestamp(), fix_time)
    except (ValueError, AttributeError):
        return (0.0, fallback_hours)

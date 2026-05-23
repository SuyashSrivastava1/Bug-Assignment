"""
src/loaders/csv_loader.py

Loads historical bug data from a local CSV file (Eclipse Bug Triaging dataset).

Expected CSV columns:
    Bug ID, Summary, Severity, Component, Assignee Real Name, Opened, Changed

Returns
-------
bugs        : list[Bug]   — all valid assigned bugs from the file.
dev_stats   : dict        — aggregated developer metrics (input to base.build_developers).
"""

import csv
import datetime
from pathlib import Path

from src.models.bug import Bug
from src.loaders.base import map_severity

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
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV dataset not found at: {filepath}")

    bugs: list[Bug] = []
    dev_stats: dict = {}

    with open(filepath, mode="r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            bug_id  = row.get("Bug ID", "").strip()
            summary = row.get("Summary", "").strip()
            assignee = (row.get("Assignee Real Name") or row.get("Assignee", "")).strip()

            # Skip rows with missing core fields
            if not bug_id or not summary or not assignee:
                continue

            severity  = map_severity(row.get("Severity", "normal"))
            component = (row.get("Component") or "general").strip()

            # Calculate fix time from timestamps
            fix_time_hours = _parse_fix_time(row.get("Opened", ""), row.get("Changed", ""))

            bug = Bug(id=bug_id, description=summary, severity=severity, module=component)
            bugs.append(bug)

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

    print(f"[CSV Loader] Loaded {len(bugs)} bugs from {filepath.name} "
          f"({len(dev_stats)} unique developers)")
    return bugs, dev_stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_fix_time(opened: str, changed: str, fallback_hours: float = 24.0) -> float:
    """Return the number of hours between two timestamp strings."""
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t0 = datetime.datetime.strptime(opened.strip(), fmt)
        t1 = datetime.datetime.strptime(changed.strip(), fmt)
        return max(0.1, (t1 - t0).total_seconds() / 3600.0)
    except (ValueError, AttributeError):
        return fallback_hours

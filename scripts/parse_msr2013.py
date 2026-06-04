"""
scripts/parse_msr2013.py

Parses the MSR 2013 Bug Dataset (v02 JSON format) and converts it into
CSV files matching the schema expected by the Bug Assignment pipeline.

Usage:
    python scripts/parse_msr2013.py

Produces:
    data/mozilla_firefox/mozilla_firefox.csv
    data/mozilla_core/mozilla_core.csv

Each CSV has the same columns as the Eclipse dataset:
    Bug ID, Product, Component, Assignee, Status, Resolution,
    Summary, Changed, Assignee Real Name, Severity, Opened
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path


# -- Paths ------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
MSR_DIR = BASE / "msr2013-bug_dataset-master" / "msr2013-bug_dataset-master" / "data" / "v02" / "mozilla"
OUTPUT_DIR = BASE / "data"


def load_json(filename: str) -> dict:
    """Load a single MSR 2013 JSON file and return the inner dict."""
    filepath = MSR_DIR / filename
    print(f"  Loading {filepath.name} ({filepath.stat().st_size / 1e6:.1f} MB)...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Each file has a single top-level key matching the field name
    key = list(data.keys())[0]
    return data[key]


def get_latest_value(history_list: list) -> str:
    """
    Given a list of {when, what, who} dicts (sorted by time),
    return the 'what' value from the last entry (most recent).
    """
    if not history_list:
        return ""
    # Sort by timestamp to be safe
    sorted_history = sorted(history_list, key=lambda x: x.get("when", 0))
    return str(sorted_history[-1].get("what", ""))


def unix_to_datetime(ts: int) -> str:
    """Convert a Unix timestamp to ISO format string."""
    try:
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return ""


def parse_and_export():
    """Main parsing logic."""
    print("=" * 60)
    print("MSR 2013 -> CSV Parser for Bug Assignment Pipeline")
    print("=" * 60)
    print(f"\nSource: {MSR_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    # -- Load all JSON files --------------------------------------------
    print("Loading JSON files...")
    reports = load_json("reports.json")
    assigned_to = load_json("assigned_to.json")
    component = load_json("component.json")
    product = load_json("product.json")
    severity = load_json("severity.json")
    short_desc = load_json("short_desc.json")
    resolution = load_json("resolution.json")
    bug_status = load_json("bug_status.json")

    all_bug_ids = set(reports.keys())
    print(f"\nTotal bugs in Mozilla dataset: {len(all_bug_ids)}")

    # -- Build flat records ---------------------------------------------
    print("\nBuilding flat records...")
    records = []

    for bug_id in all_bug_ids:
        report = reports[bug_id]

        # Get the latest value for each field
        prod = get_latest_value(product.get(bug_id, []))
        comp = get_latest_value(component.get(bug_id, []))
        sev = get_latest_value(severity.get(bug_id, []))
        desc = get_latest_value(short_desc.get(bug_id, []))
        res = get_latest_value(resolution.get(bug_id, []))
        status = get_latest_value(bug_status.get(bug_id, []))

        # Get the last assignee (the person who ultimately resolved it)
        assignee_history = assigned_to.get(bug_id, [])
        assignee = get_latest_value(assignee_history)

        # Get opening timestamp
        opening_ts = report.get("opening", 0)
        opened = unix_to_datetime(opening_ts)

        # Get the last change timestamp (from assignee or status history)
        all_timestamps = []
        for field_data in [assignee_history,
                           bug_status.get(bug_id, []),
                           resolution.get(bug_id, [])]:
            if field_data:
                for entry in field_data:
                    all_timestamps.append(entry.get("when", 0))
        changed = unix_to_datetime(max(all_timestamps)) if all_timestamps else opened

        records.append({
            "Bug ID": int(bug_id),
            "Product": prod,
            "Component": comp,
            "Assignee": assignee,
            "Status": status,
            "Resolution": res,
            "Summary": desc,
            "Changed": changed,
            "Assignee Real Name": assignee,  # Email is the best we have
            "Severity": sev,
            "Opened": opened,
        })

    print(f"Total records built: {len(records)}")

    # -- Count products -------------------------------------------------
    product_counts = {}
    for r in records:
        p = r["Product"]
        product_counts[p] = product_counts.get(p, 0) + 1

    print("\nProduct distribution:")
    for p, count in sorted(product_counts.items(), key=lambda x: -x[1]):
        print(f"  {p}: {count}")

    # -- Filter and export by product -----------------------------------
    PRODUCT_MAP = {
        "Firefox": ("mozilla_firefox", "Firefox"),
        "Core": ("mozilla_core", "Core"),
        "Thunderbird": ("thunderbird", "Thunderbird"),
    }

    for product_name, (folder_name, display_name) in PRODUCT_MAP.items():
        # Filter for this product + FIXED resolution only
        filtered = [r for r in records
                    if r["Product"] == product_name
                    and r["Resolution"] == "FIXED"]

        if not filtered:
            print(f"\n[!] No FIXED bugs found for product '{product_name}', skipping.")
            continue

        # Sort by opening date (chronological)
        filtered.sort(key=lambda x: x["Opened"])

        # Remove bugs with empty assignees or "nobody" assignees
        filtered = [r for r in filtered
                    if r["Assignee"]
                    and "nobody" not in r["Assignee"].lower()
                    and r["Assignee"] != "{}"]

        # Count unique developers
        devs = set(r["Assignee"] for r in filtered)

        print(f"\n{'-' * 50}")
        print(f"Product: {display_name}")
        print(f"  Total FIXED bugs (with real assignees): {len(filtered)}")
        print(f"  Unique developers: {len(devs)}")
        print(f"  Date range: {filtered[0]['Opened']} -> {filtered[-1]['Opened']}")

        # -- Export to CSV ----------------------------------------------
        out_dir = OUTPUT_DIR / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{folder_name}.csv"

        fieldnames = [
            "Bug ID", "Product", "Component", "Assignee", "Status",
            "Resolution", "Summary", "Changed", "Assignee Real Name",
            "Severity", "Opened"
        ]

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(filtered)

        size_mb = out_file.stat().st_size / 1e6
        print(f"  Exported: {out_file} ({size_mb:.2f} MB)")

    print(f"\n{'=' * 60}")
    print("Done! CSV files are ready in the data/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    parse_and_export()

"""
src/loaders/bugzilla_loader.py

Loads bug description text from the Bugzilla corpus flat-text file
(e.g. the qicongliu/bugzilla-bug-reports dataset: "corpus (fixsev).txt").

File format
-----------
The corpus file is a semi-structured log where each bug entry looks like:

    Wednesday April 29 1998 2:46:32 AM PDT
    Updated by: john.doe@mozilla.org
    Assign to: dev.helper@mozilla.org

    Yes the combo box sucks and has lots of bugs.  I believe I've tracked
    this bug down to the following...

Entries are separated by one or more blank lines.  Metadata lines begin
with recognisable prefixes (timestamps, "Updated by:", "Assign to:", etc.)
and are excluded from the extracted description.

Usage in the pipeline
---------------------
This loader returns (bugs, {}) — an empty dev_stats dict — because the
corpus is used **as NLP context only**, exactly like the Eclipse CSV is used
when routing GitHub bugs.  No Bugzilla developers are ever added to the
candidate pool.

Returns
-------
bugs      : list[Bug]   — description text extracted from each corpus entry.
dev_stats : dict        — always empty (context-only source).
"""

import re
from pathlib import Path

from src.models.bug import Bug
from src.loaders.base import map_severity

# Default path relative to the project root
DEFAULT_BUGZILLA_PATH = Path("data") / "bugzilla" / "corpus (fixsev).txt"

# Regex patterns that identify metadata / header lines we skip
_META_PATTERNS = [
    re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", re.I),
    re.compile(r"^Updated by\s*:", re.I),
    re.compile(r"^Assign(ed)? to\s*:", re.I),
    re.compile(r"^Reported by\s*:", re.I),
    re.compile(r"^Priority\s*:", re.I),
    re.compile(r"^Severity\s*:", re.I),
    re.compile(r"^Status\s*:", re.I),
    re.compile(r"^Resolution\s*:", re.I),
    re.compile(r"^Component\s*:", re.I),
    re.compile(r"^Product\s*:", re.I),
    re.compile(r"^Version\s*:", re.I),
    re.compile(r"^Hardware\s*:", re.I),
    re.compile(r"^OS\s*:", re.I),
    re.compile(r"^Summary\s*:", re.I),
    re.compile(r"^CC\s*:", re.I),
    re.compile(r"^Keywords\s*:", re.I),
    re.compile(r"^URL\s*:", re.I),
    re.compile(r"^Bug\s+\d+", re.I),
    # Email addresses on their own line
    re.compile(r"^[\w.+\-]+@[\w.\-]+\.\w{2,}$"),
    # Pure timestamp / date-like lines
    re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"),
]

# Keywords used to infer severity from a block's header text
_CRITICAL_KW = {"blocker", "critical", "crash", "major", "showstopper", "p1", "p0"}
_MINOR_KW    = {"trivial", "minor", "enhancement", "feature", "request", "p4", "p5"}


def load(filepath: str | Path = DEFAULT_BUGZILLA_PATH) -> tuple[list[Bug], dict]:
    """
    Parse the Bugzilla corpus file and return Bug objects for NLP context.

    Parameters
    ----------
    filepath : Path to the corpus .txt file.

    Returns
    -------
    (bugs, {})  — dev_stats is always empty; this source is context-only.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"[Bugzilla Loader] File not found at {filepath} — skipping.")
        return [], {}

    bugs: list[Bug] = []
    block_lines: list[str] = []
    bug_counter = 0

    def _flush_block(block: list[str], counter: int) -> Bug | None:
        """Convert one collected block of lines into a Bug, or None if empty."""
        header_text = " ".join(block[:6]).lower()  # inspect first 6 lines for severity

        desc_parts = []
        for line in block:
            line = line.strip()
            if not line:
                continue
            if any(pat.match(line) for pat in _META_PATTERNS):
                continue
            desc_parts.append(line)

        description = " ".join(desc_parts).strip()
        if len(description) < 10:   # skip near-empty blocks
            return None

        # Infer severity from header keywords
        words = set(re.findall(r"\w+", header_text))
        if words & _CRITICAL_KW:
            severity = "critical"
        elif words & _MINOR_KW:
            severity = "minor"
        else:
            severity = "normal"

        return Bug(
            id=f"bzl_{counter}",
            description=description,
            severity=severity,
            module="general",
        )

    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")

                # A blank line is a block separator
                if line.strip() == "":
                    if block_lines:
                        bug_counter += 1
                        bug = _flush_block(block_lines, bug_counter)
                        if bug is not None:
                            bugs.append(bug)
                        block_lines = []
                else:
                    block_lines.append(line)

        # Flush any trailing block at EOF
        if block_lines:
            bug_counter += 1
            bug = _flush_block(block_lines, bug_counter)
            if bug is not None:
                bugs.append(bug)

    except OSError as exc:
        print(f"[Bugzilla Loader] Could not read file: {exc}")
        return [], {}

    print(
        f"[Bugzilla Loader] Loaded {len(bugs):,} bug descriptions "
        f"from {filepath.name}"
    )
    return bugs, {}

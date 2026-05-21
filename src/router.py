"""
src/router.py

The Router is the central coordinator of the pipeline.

It is responsible for:
  1. Loading the offline Eclipse CSV as background context (improves AI accuracy).
  2. Loading live GitHub data for a specific repository (determines the candidate pool).
  3. Merging the two data sources correctly:
       - History  : combined (Eclipse + GitHub)  → better NLP similarity matching
       - Candidates: GitHub developers ONLY       → results are always project-specific
  4. Training the BugAssigner.
  5. Accepting a new bug and returning ranked developer assignments.
"""

import re
from pathlib import Path

from src.models.bug import Bug
from src.models.assigner import BugAssigner
from src.loaders import csv_loader, github_loader
from src.loaders.base import build_developers, map_severity

# Path to the offline Eclipse dataset used as background training context
_ECLIPSE_CSV = Path("archive") / "final dataset for work ecllipse.csv"


class Router:
    """
    End-to-end pipeline coordinator.

    Example
    -------
    router = Router()
    router.train_from_github("microsoft/vscode")
    best, rankings = router.assign(my_bug)
    """

    def __init__(self):
        self._assigner = BugAssigner()
        self._project_developers = []
        self._trained = False

    # ------------------------------------------------------------------
    # Training modes
    # ------------------------------------------------------------------

    def train_from_github(self, repo: str, limit: int = 100) -> None:
        """
        Build the model using a GitHub repository as the candidate source.

        The offline Eclipse dataset is added transparently as training context
        to improve NLP accuracy — but Eclipse developers are never included as
        candidates.

        Parameters
        ----------
        repo  : "owner/repo", e.g. "facebook/react"
        limit : Number of recent closed issues/PRs to fetch from GitHub.
        """
        print(f"\n[Router] Loading project history from GitHub: {repo}")
        github_bugs, github_dev_stats = github_loader.load(repo, limit=limit)

        if not github_bugs:
            raise ValueError(
                f"No resolved bugs with identifiable developers found in {repo}.\n"
                "This usually means the project uses a different workflow "
                "(e.g. no assignees and no PRs). Try a different repo."
            )

        # Build the project-specific candidate pool
        project_developers, project_resolutions = build_developers(github_dev_stats)
        self._project_developers = project_developers

        # Load Eclipse bugs as background NLP context (no developers extracted)
        print("[Router] Loading offline Eclipse dataset for NLP context...")
        try:
            eclipse_bugs, _ = csv_loader.load(_ECLIPSE_CSV)
        except FileNotFoundError:
            print("[Router] Eclipse dataset not found — continuing without offline context.")
            eclipse_bugs = []

        # Combined history: Eclipse (context) + GitHub (context + resolutions)
        combined_history = eclipse_bugs + github_bugs

        print(f"[Router] Training on {len(combined_history)} bugs "
              f"| Candidate pool: {len(project_developers)} developers from {repo}")

        self._assigner.fit(combined_history, project_developers, project_resolutions)
        self._trained = True

    def train_from_csv(self, filepath: str | Path = _ECLIPSE_CSV) -> None:
        """
        Build the model using a local CSV file as both context and candidate source.

        Parameters
        ----------
        filepath : Path to the CSV file.
        """
        print(f"\n[Router] Loading dataset from CSV: {filepath}")
        bugs, dev_stats = csv_loader.load(filepath)
        developers, resolutions = build_developers(dev_stats)
        self._project_developers = developers

        print(f"[Router] Training on {len(bugs)} bugs "
              f"| Candidate pool: {len(developers)} developers")
        self._assigner.fit(bugs, developers, resolutions)
        self._trained = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def assign(self, bug: Bug, k: int = 5) -> tuple:
        """
        Assign a bug to the most suitable developer.

        Parameters
        ----------
        bug : A Bug object to assign.
        k   : Number of nearest historical neighbours to use as candidate source.

        Returns
        -------
        (best_developer, ranked_list, weights)
        ranked_list is a list of (Developer, score) tuples sorted best-first.
        Returns (None, [], None) if no assignment can be made.
        """
        if not self._trained:
            raise RuntimeError("Router has not been trained. Call train_from_github() or train_from_csv() first.")
        return self._assigner.assign(bug, k=k)


# ---------------------------------------------------------------------------
# GitHub URL helpers (used by main.py)
# ---------------------------------------------------------------------------

def parse_github_issue_url(url: str) -> tuple[str | None, str | None]:
    """
    Extract (owner/repo, issue_number) from a GitHub issue URL.

    Accepts:
      https://github.com/microsoft/vscode/issues/212260
      github.com/facebook/react/issues/36469

    Returns (None, None) for invalid URLs.
    """
    match = re.search(r"github\.com/([^/]+/[^/]+)/issues/(\d+)", url)
    if match:
        return match.group(1), match.group(2)
    return None, None


def build_bug_from_github_issue(repo: str, issue_number: str | int) -> Bug | None:
    """
    Fetch a single GitHub issue and convert it to a Bug object.

    Returns None if the issue cannot be fetched.
    """
    data = github_loader.fetch_single_issue(repo, issue_number)
    if not data or "title" not in data:
        return None

    title    = data.get("title", "")
    body     = data.get("body") or ""
    labels   = [lbl["name"].lower() for lbl in data.get("labels", [])]
    severity = github_loader._severity_from_labels(labels)
    module   = github_loader._component_from_labels(labels)

    return Bug(
        id=int(issue_number),
        description=f"{title} {body}".strip(),
        severity=severity,
        module=module,
    )

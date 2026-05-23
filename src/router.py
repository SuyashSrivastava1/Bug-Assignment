"""
src/router.py

The Router is the central coordinator of the pipeline.

It is responsible for:
  1. Loading offline NLP context datasets (Eclipse CSV + Bugzilla corpus) to
     improve AI accuracy.
  2. Loading live GitHub data for a specific repository (determines the
     candidate pool).
  3. Merging the data sources correctly:
       - History  : all context sources combined  -> better NLP similarity
       - Candidates: GitHub developers ONLY        -> always project-specific
  4. Training the BugAssigner.
  5. Accepting a new bug and returning ranked developer assignments.

NLP Context Sources
-------------------
Both offline datasets improve the TF-IDF vocabulary without polluting the
candidate pool.  They are transparent to the caller — loaded automatically
whenever training begins.

  data/eclipse/final dataset for work ecllipse.csv
      10,000 structured Eclipse bug reports (Bug ID, Summary, Severity, ...).

  data/bugzilla/corpus (fixsev).txt
      35,000+ Bugzilla log-format entries from 50+ open-source projects.
      Download from: kaggle.com/datasets/qicongliu/bugzilla-bug-reports
"""

import re
from pathlib import Path

from src.models.bug import Bug
from src.models.assigner import BugAssigner
from src.loaders import csv_loader, github_loader, bugzilla_loader
from src.loaders.base import build_developers, map_severity

# ---------------------------------------------------------------------------
# Paths to offline NLP context datasets (relative to project root)
# ---------------------------------------------------------------------------

_ECLIPSE_CSV   = Path("data") / "eclipse" / "final dataset for work ecllipse.csv"
_BUGZILLA_TXT  = Path("data") / "bugzilla" / "corpus (fixsev).txt"


class Router:
    """
    End-to-end pipeline coordinator.

    Example
    -------
    router = Router()
    router.train_from_github("microsoft/vscode")
    best, rankings, weights, signals = router.assign(my_bug)
    """

    def __init__(self):
        self._assigner = BugAssigner()
        self._project_developers = []
        self._trained = False

    # ------------------------------------------------------------------
    # NLP context loader (shared by all training modes)
    # ------------------------------------------------------------------

    def _load_nlp_context(self) -> list[Bug]:
        """
        Load all offline NLP context datasets.

        Returns a combined list of Bug objects that will be fed to the
        TF-IDF vectoriser to enrich its vocabulary.  No developer objects
        are built from these sources.
        """
        context_bugs: list[Bug] = []

        # -- Eclipse CSV --
        print("[Router] Loading Eclipse dataset for NLP context...")
        try:
            eclipse_bugs, _ = csv_loader.load(_ECLIPSE_CSV)
            context_bugs.extend(eclipse_bugs)
        except FileNotFoundError:
            print("[Router] Eclipse dataset not found — skipping.")

        # -- Bugzilla corpus --
        print("[Router] Loading Bugzilla corpus for NLP context...")
        bugzilla_bugs, _ = bugzilla_loader.load(_BUGZILLA_TXT)
        context_bugs.extend(bugzilla_bugs)

        total = len(context_bugs)
        if total:
            print(f"[Router] NLP context: {total:,} bug descriptions total "
                  f"(Eclipse {len(eclipse_bugs) if 'eclipse_bugs' in dir() else 0:,} "
                  f"+ Bugzilla {len(bugzilla_bugs):,})")
        else:
            print("[Router] Warning: no NLP context loaded — accuracy may be reduced.")

        return context_bugs

    # ------------------------------------------------------------------
    # Training modes
    # ------------------------------------------------------------------

    def train_from_github(self, repo: str, limit: int = 100) -> None:
        """
        Build the model using a GitHub repository as the candidate source.

        Offline NLP context (Eclipse + Bugzilla) is loaded automatically to
        improve TF-IDF accuracy.  These developers are never included as
        candidates — only contributors from the target repo are.

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

        # Load all offline NLP context (Eclipse + Bugzilla)
        context_bugs = self._load_nlp_context()

        # Combined history: context first, then project bugs
        combined_history = context_bugs + github_bugs

        print(f"[Router] Training on {len(combined_history):,} bugs total "
              f"| Candidate pool: {len(project_developers)} developers from {repo}")

        self._assigner.fit(combined_history, project_developers, project_resolutions)
        self._trained = True

    def train_from_csv(self, filepath: str | Path = _ECLIPSE_CSV) -> None:
        """
        Build the model using a local Eclipse CSV as both context and
        candidate source.  The Bugzilla corpus is also loaded as extra NLP
        context to improve vocabulary quality.

        Parameters
        ----------
        filepath : Path to the Eclipse CSV file.
        """
        print(f"\n[Router] Loading dataset from CSV: {filepath}")
        bugs, dev_stats = csv_loader.load(filepath)
        developers, resolutions = build_developers(dev_stats)
        self._project_developers = developers

        # Load Bugzilla as additional NLP context
        print("[Router] Loading Bugzilla corpus for extra NLP context...")
        bugzilla_bugs, _ = bugzilla_loader.load(_BUGZILLA_TXT)

        combined_history = bugzilla_bugs + bugs

        print(f"[Router] Training on {len(combined_history):,} bugs "
              f"(Bugzilla context {len(bugzilla_bugs):,} + Eclipse {len(bugs):,}) "
              f"| Candidate pool: {len(developers)} developers")

        self._assigner.fit(combined_history, developers, resolutions)
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
        (best_developer, ranked_list, weights, signals)
        ranked_list is a list of (Developer, score) tuples sorted best-first.
        weights is the per-bug dynamic weight vector (unique for every bug).
        signals is a dict of the NLP signals that drove the weights.
        Returns (None, [], None, {}) if no assignment can be made.
        """
        if not self._trained:
            raise RuntimeError(
                "Router has not been trained. "
                "Call train_from_github() or train_from_csv() first."
            )
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

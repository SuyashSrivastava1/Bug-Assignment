"""
src/models/assigner.py

The core Bug-to-Developer Assignment algorithm.

Pipeline (7 steps):
  1. Feature Extraction     — TF-IDF vectorise the new bug description.
  2. Similarity Search      — KNN cosine similarity against historical bugs.
  3. Candidate Extraction   — Identify developers who fixed the top-K similar bugs.
  4. Hard Filtering         — Remove developers on leave or over 95% workload.
  5. Attribute Matrix       — Build a matrix of the 5 performance metrics.
  6. Dynamic Weighting      — Compute per-bug weights from NLP text signals (NOT a preset).
  7. WSM Normalisation      — Min-Max scale each column, compute weighted utility scores, rank.

Dynamic Weighting (Step 6)
--------------------------
Every bug gets a UNIQUE weight vector computed fresh from its description text.
Three independent signal scores are extracted:

  Urgency Score    — "crash", "outage", "production", "security", etc.
                     High urgency → heavier weight on Fix Time and Domain Skill.

  Complexity Score — "memory leak", "threading", "algorithm", "architecture", etc.
                     High complexity → heavier weight on Experience and Success Rate.

  Routine Score    — "typo", "css", "alignment", "cosmetic", "padding", etc.
                     High routine → heavier weight on Workload (assign to someone free).

Severity (critical / normal / minor) acts as a continuous amplifier of the urgency
signal — it does NOT pick from a fixed table.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.bug import Bug
from src.models.developer import Developer


# ---------------------------------------------------------------------------
# NLP Signal Keyword Sets
# ---------------------------------------------------------------------------

URGENCY_KEYWORDS = {
    "crash", "crashed", "crashing", "down", "outage", "production", "prod",
    "hotfix", "emergency", "regression", "breaking", "broken", "blocker",
    "data loss", "security", "vulnerability", "exploit", "breach",
    "freeze", "unresponsive", "hang", "hangs", "infinite loop",
    "null pointer", "segfault", "exception", "500", "404", "403",
    "urgent", "critical", "showstopper", "p0", "p1",
}

COMPLEXITY_KEYWORDS = {
    "memory", "leak", "threading", "concurrent", "concurrency",
    "race condition", "deadlock", "algorithm", "performance",
    "architecture", "refactor", "scalability", "bottleneck",
    "optimization", "async", "synchronization", "distributed",
    "database", "migration", "cache", "caching", "integration",
    "latency", "throughput", "indexing", "sharding", "replication",
    "cryptography", "encryption", "authentication", "oauth",
    "dependency", "circular", "heap", "stack overflow",
}

ROUTINE_KEYWORDS = {
    "typo", "alignment", "color", "colour", "style", "cosmetic",
    "spelling", "css", "hover", "padding", "margin", "font",
    "icon", "tooltip", "wording", "label", "documentation",
    "comment", "whitespace", "lint", "formatting", "indent",
    "minor", "trivial", "small", "simple", "rename", "cleanup",
    "translation", "i18n", "l10n", "placeholder", "readme",
}

# Column indices (matches attribute matrix order)
BENEFIT_COLS = {0, 2, 4}   # Experience, Success Rate, Domain Skill
COST_COLS    = {1, 3}       # Fix Time, Workload


# ---------------------------------------------------------------------------
# Per-bug Dynamic Weight Engine
# ---------------------------------------------------------------------------

def _count_signal(text: str, keywords: set) -> int:
    """Count how many keyword phrases appear in text."""
    count = 0
    for kw in keywords:
        if kw in text:
            count += 1
    return count


def compute_dynamic_weights(bug: Bug) -> tuple:
    """
    Compute a unique weight vector for this specific bug based on NLP text signals.

    The weight vector has 5 components (columns match the attribute matrix):
      [Experience, Fix Time, Success Rate, Workload, Domain Skill]

    Returns
    -------
    (weights, signals)
    weights : np.ndarray  — normalised to sum exactly to 1.0
    signals : dict        — breakdown of detected signals for display / debugging
    """
    text = (bug.description + " " + bug.severity).lower()

    # --- Signal extraction ---
    urgency_hits    = {kw for kw in URGENCY_KEYWORDS    if kw in text}
    complexity_hits = {kw for kw in COMPLEXITY_KEYWORDS if kw in text}
    routine_hits    = {kw for kw in ROUTINE_KEYWORDS    if kw in text}

    # Normalise hit counts to [0, 1]; cap at 3 hits → score 1.0
    urgency_score    = min(len(urgency_hits)    / 3.0, 1.0)
    complexity_score = min(len(complexity_hits) / 3.0, 1.0)
    routine_score    = min(len(routine_hits)    / 3.0, 1.0)

    # Severity as a continuous factor: critical=1.0, normal=0.5, minor=0.0
    severity_score = {"critical": 1.0, "normal": 0.5, "minor": 0.0}.get(bug.severity, 0.5)

    # Combined urgency pressure: severity amplifies urgency signal
    effective_urgency = min(
        urgency_score * 0.5 + severity_score * 0.5 + urgency_score * severity_score * 0.3,
        1.0,
    )

    # --- Raw weight formula ---
    # Each weight has a minimum base of 0.10 so no dimension is ever ignored.
    # Signals add on top of that base.
    w_experience  = 0.10 + 0.20 * complexity_score + 0.05 * severity_score
    w_fixtime     = 0.10 + 0.40 * effective_urgency
    w_successrate = 0.10 + 0.15 * complexity_score + 0.05 * severity_score
    w_workload    = 0.10 + 0.50 * routine_score + 0.15 * max(0.0, 1.0 - effective_urgency - complexity_score)
    w_domainskill = 0.10 + 0.25 * severity_score + 0.15 * urgency_score

    raw = np.array([w_experience, w_fixtime, w_successrate, w_workload, w_domainskill],
                   dtype=float)

    # Normalise so weights sum to exactly 1.0
    weights = raw / raw.sum()

    signals = {
        "urgency_score":    round(float(urgency_score),    3),
        "complexity_score": round(float(complexity_score), 3),
        "routine_score":    round(float(routine_score),    3),
        "severity_score":   round(float(severity_score),   3),
        "effective_urgency": round(float(effective_urgency), 3),
        "urgency_keywords":    sorted(urgency_hits),
        "complexity_keywords": sorted(complexity_hits),
        "routine_keywords":    sorted(routine_hits),
    }

    return weights, signals


# ---------------------------------------------------------------------------
# BugAssigner
# ---------------------------------------------------------------------------

class BugAssigner:
    """
    Trains on historical bug-resolution data and assigns new bugs to developers.

    Usage
    -----
    assigner = BugAssigner()
    assigner.fit(historical_bugs, developers, resolutions)
    best_dev, rankings, weights, signals = assigner.assign(new_bug, k=5)
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self._historical_bugs: list[Bug] = []
        self._historical_vectors = None
        self._developers: list[Developer] = []
        self._resolutions: dict = {}  # bug_id -> developer_id

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        historical_bugs: list[Bug],
        developers: list[Developer],
        resolutions: dict,
    ) -> None:
        """
        Fit the TF-IDF model and store all training data.

        Parameters
        ----------
        historical_bugs : list of Bug objects used as the knowledge base.
        developers      : list of Developer objects — the candidate pool.
        resolutions     : dict mapping bug.id -> developer.id (who fixed what).
        """
        self._historical_bugs = historical_bugs
        self._developers = developers
        self._resolutions = resolutions

        corpus = [bug.description for bug in historical_bugs]
        self._historical_vectors = self._vectorizer.fit_transform(corpus).toarray()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def assign(self, new_bug: Bug, k: int = 5) -> tuple:
        """
        Assign a new bug to the most suitable developer.

        Returns
        -------
        (best_developer, ranked_list, weights, signals)

        best_developer : Developer | None
        ranked_list    : list of (Developer, score) tuples, best first
        weights        : np.ndarray — the unique weight vector used for this bug
        signals        : dict — NLP signals that drove the weight calculation

        Returns (None, [], None, {}) if no assignment can be made.
        """
        if self._historical_vectors is None:
            raise RuntimeError("BugAssigner has not been trained yet. Call fit() first.")

        # Step 1 — Feature Extraction
        new_vector = self._vectorizer.transform([new_bug.description]).toarray()[0]

        # Step 2 — KNN Similarity Search
        similarities = cosine_similarity([new_vector], self._historical_vectors)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]

        # Step 3 — Candidate Extraction
        candidate_ids = set()
        for idx in top_k_indices:
            bug = self._historical_bugs[idx]
            if bug.id in self._resolutions:
                candidate_ids.add(self._resolutions[bug.id])

        candidates = [d for d in self._developers if d.id in candidate_ids]

        # Step 4 — Hard Filtering (availability + workload cap)
        available = [d for d in candidates if not d.on_leave and d.workload <= 95]

        if not available:
            # Fallback: open pool to all available developers
            available = [d for d in self._developers if not d.on_leave and d.workload <= 95]

        if not available:
            return None, [], None, {}

        # Step 5 — Attribute Matrix
        # Columns: [Experience, Fix Time, Success Rate, Workload, Domain Skill]
        matrix = np.array(
            [[d.experience, d.fix_time, d.success_rate, d.workload, d.domain_skill]
             for d in available],
            dtype=float,
        )

        # Step 6 — Per-Bug Dynamic Weight Computation (unique for every bug)
        weights, signals = compute_dynamic_weights(new_bug)

        # Step 7 — Min-Max Normalisation + WSM
        mins = matrix.min(axis=0)
        maxs = matrix.max(axis=0)
        norm = np.zeros_like(matrix)

        for j in range(matrix.shape[1]):
            rng = maxs[j] - mins[j]
            if rng == 0:
                norm[:, j] = 1.0  # All equal -> full score for everyone
            elif j in BENEFIT_COLS:
                norm[:, j] = (matrix[:, j] - mins[j]) / rng
            else:  # Cost column
                norm[:, j] = (maxs[j] - matrix[:, j]) / rng

        scores = norm @ weights
        order = scores.argsort()[::-1]
        ranked = [(available[i], float(scores[i])) for i in order]

        return ranked[0][0], ranked, weights, signals

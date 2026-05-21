"""
src/models/assigner.py

The core Bug-to-Developer Assignment algorithm.

Pipeline (7 steps):
  1. Feature Extraction     — TF-IDF vectorise the new bug description.
  2. Similarity Search      — KNN cosine similarity against historical bugs.
  3. Candidate Extraction   — Identify developers who fixed the top-K similar bugs.
  4. Hard Filtering         — Remove developers on leave or over 95% workload.
  5. Attribute Matrix       — Build a matrix of the 5 performance metrics.
  6. Dynamic Weighting      — Assign criteria weights based on bug severity.
  7. WSM Normalisation      — Min-Max scale each column, compute weighted utility scores, rank.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.bug import Bug
from src.models.developer import Developer


# ---------------------------------------------------------------------------
# Weight presets  (Experience, Fix Time, Success Rate, Workload, Domain Skill)
# Weights sum to 1.0. Fix Time and Workload are Cost criteria (lower is better).
# ---------------------------------------------------------------------------
WEIGHTS = {
    "critical": np.array([0.10, 0.40, 0.10, 0.10, 0.30]),  # Speed + Skill
    "minor":    np.array([0.10, 0.10, 0.10, 0.60, 0.10]),  # Low workload
    "normal":   np.array([0.20, 0.20, 0.20, 0.20, 0.20]),  # Balanced
}

# Column indices that are Benefit criteria (higher → better)
BENEFIT_COLS = {0, 2, 4}   # Experience, Success Rate, Domain Skill
# Column indices that are Cost criteria (lower → better)
COST_COLS    = {1, 3}       # Fix Time, Workload


class BugAssigner:
    """
    Trains on historical bug-resolution data and assigns new bugs to developers.

    Usage
    -----
    assigner = BugAssigner()
    assigner.fit(historical_bugs, developers, resolutions)
    best_dev, rankings = assigner.assign(new_bug, k=5)
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
        (best_developer, ranked_list, weights)
        ranked_list is a list of (Developer, score) tuples, best first.
        Returns (None, [], None) if no assignment can be made.
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
            return None, [], None

        # Step 5 — Attribute Matrix
        # Columns: [Experience, Fix Time, Success Rate, Workload, Domain Skill]
        matrix = np.array(
            [[d.experience, d.fix_time, d.success_rate, d.workload, d.domain_skill]
             for d in available],
            dtype=float,
        )

        # Step 6 — Dynamic Weights
        weights = WEIGHTS.get(new_bug.severity, WEIGHTS["normal"])

        # Step 7 — Min-Max Normalisation + WSM
        mins = matrix.min(axis=0)
        maxs = matrix.max(axis=0)
        norm = np.zeros_like(matrix)

        for j in range(matrix.shape[1]):
            rng = maxs[j] - mins[j]
            if rng == 0:
                norm[:, j] = 1.0  # All equal → full score for everyone
            elif j in BENEFIT_COLS:
                norm[:, j] = (matrix[:, j] - mins[j]) / rng
            else:  # Cost column
                norm[:, j] = (maxs[j] - matrix[:, j]) / rng

        scores = norm @ weights
        order = scores.argsort()[::-1]
        ranked = [(available[i], float(scores[i])) for i in order]

        return ranked[0][0], ranked, weights

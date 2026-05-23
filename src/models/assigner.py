"""
src/models/assigner.py

The core Bug-to-Developer Assignment algorithm.

Pipeline (8 steps):
  1. Feature Extraction     — TF-IDF vectorise the new bug description (sparse, bigrams).
  2. Similarity Search      — KNN cosine similarity against all historical bugs (K=20).
  3. Candidate Extraction   — Identify developers who fixed the top-K similar bugs.
  4. Hard Filtering         — Remove developers on leave or over 95% workload.
  5. KNN Affinity           — Per-developer topical expertise score: mean cosine similarity
                              between the new bug and every bug that developer has fixed.
  6. Attribute Matrix       — Build a matrix of 6 performance metrics per candidate.
  7. Dynamic Weighting      — Compute per-bug weights from NLP text signals (NOT a preset).
  8. WSM + Component Bonus  — Min-Max scale, weighted sum, apply component-match multiplier.

Accuracy Improvements (v2)
--------------------------
Five changes over v1 to increase Top-1/Top-3/Top-5:

  1. TF-IDF tuning: bigrams (ngram_range=(1,2)), sublinear_tf=True, max_features=50_000.
     Bigrams like "memory leak", "null pointer", "race condition" are highly discriminative.
     Sublinear TF dampens the effect of very frequent terms.

  2. Wider candidate pool: K raised from 5 to 20. With only 5 neighbours the correct
     developer was often never in the candidate set — a hard ceiling on accuracy.

  3. KNN Affinity (6th metric): for every candidate, compute the average cosine similarity
     between the new bug and all historical bugs that developer has previously fixed.
     This directly measures topical expertise — not just global experience.

  4. Component match bonus: if the new bug's module matches any component in a developer's
     history, their final score is multiplied by 1.1 (a 10% boost).

  5. Sparse vectors: vectors stay as scipy sparse CSR matrices instead of being densified,
     making the approach memory-safe with 200k+ training bugs.

Dynamic Weighting (Step 7)
--------------------------
Every bug gets a UNIQUE 6-element weight vector computed fresh from its description text.
Three independent signal scores drive the weights:

  Urgency Score    — "crash", "outage", "production", "security", etc.
                     High urgency → heavier weight on Fix Time and Domain Skill.

  Complexity Score — "memory leak", "threading", "algorithm", "architecture", etc.
                     High complexity → heavier weight on Experience, Success Rate, and KNN Affinity.

  Routine Score    — "typo", "css", "alignment", "cosmetic", "padding", etc.
                     High routine → heavier weight on Workload (assign to someone free).

Severity (critical / normal / minor) acts as a continuous amplifier of the urgency
signal — it does NOT pick from a fixed table.
"""

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

# Column indices in the 6-column attribute matrix
# [Experience, Fix Time, Success Rate, Workload, Domain Skill, KNN Affinity]
BENEFIT_COLS = {0, 2, 4, 5}   # Experience, Success Rate, Domain Skill, KNN Affinity
COST_COLS    = {1, 3}          # Fix Time, Workload


# ---------------------------------------------------------------------------
# Per-bug Dynamic Weight Engine
# ---------------------------------------------------------------------------

def compute_dynamic_weights(bug: Bug) -> tuple:
    """
    Compute a unique 6-element weight vector for this bug based on NLP text signals.

    The weight vector columns match the attribute matrix:
      [Experience, Fix Time, Success Rate, Workload, Domain Skill, KNN Affinity]

    Returns
    -------
    (weights, signals)
    weights : np.ndarray  — 6 elements, normalised to sum exactly to 1.0
    signals : dict        — breakdown of detected signals for display / debugging
    """
    text = (bug.description + " " + bug.severity).lower()

    # --- Signal extraction ---
    urgency_hits    = {kw for kw in URGENCY_KEYWORDS    if kw in text}
    complexity_hits = {kw for kw in COMPLEXITY_KEYWORDS if kw in text}
    routine_hits    = {kw for kw in ROUTINE_KEYWORDS    if kw in text}

    # Normalise hit counts to [0, 1]; cap at 3 hits -> score 1.0
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

    # --- Raw weight formula (6 dimensions) ---
    # Each weight has a minimum base so no dimension is ever completely ignored.
    # Signals add on top of the base.
    w_experience  = 0.10 + 0.20 * complexity_score + 0.05 * severity_score
    w_fixtime     = 0.10 + 0.40 * effective_urgency
    w_successrate = 0.10 + 0.15 * complexity_score + 0.05 * severity_score
    w_workload    = 0.10 + 0.50 * routine_score + 0.15 * max(0.0, 1.0 - effective_urgency - complexity_score)
    w_domainskill = 0.10 + 0.25 * severity_score + 0.15 * urgency_score
    # KNN Affinity: always meaningful; boosted by complexity (need topical expert)
    # and slightly by urgency (want someone who's fixed similar urgent bugs before)
    w_knnaffinity = 0.15 + 0.15 * complexity_score + 0.10 * effective_urgency

    raw = np.array(
        [w_experience, w_fixtime, w_successrate, w_workload, w_domainskill, w_knnaffinity],
        dtype=float,
    )

    # Normalise so weights sum to exactly 1.0
    weights = raw / raw.sum()

    signals = {
        "urgency_score":     round(float(urgency_score),     3),
        "complexity_score":  round(float(complexity_score),  3),
        "routine_score":     round(float(routine_score),     3),
        "severity_score":    round(float(severity_score),    3),
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
    best_dev, rankings, weights, signals = assigner.assign(new_bug, k=20)
    """

    def __init__(self):
        # TF-IDF with bigrams, sublinear TF, and capped vocabulary.
        # Stays sparse — never call .toarray() on the full matrix.
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,           # include all terms (small unit-test corpora need this)
            max_df=0.95,        # drop terms appearing in >95% of documents (near-stop-words)
            ngram_range=(1, 2), # unigrams + bigrams ("memory leak", "null pointer")
            max_features=50_000,
            sublinear_tf=True,  # use log(1 + tf) — dampens very frequent terms
        )
        self._historical_bugs: list[Bug] = []
        self._historical_vectors = None          # scipy CSR sparse matrix
        self._developers: list[Developer] = []
        self._resolutions: dict = {}             # bug_id -> developer_id

        # Per-developer lookup tables built at fit() time
        self._dev_bug_indices: dict[int, list[int]] = {}  # dev_id -> [hist indices]
        self._dev_components: dict[int, set[str]]   = {}  # dev_id -> {module, ...}

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
        Fit the TF-IDF model and build per-developer lookup tables.

        Parameters
        ----------
        historical_bugs : All Bug objects (NLP context + project bugs) for the vectoriser.
        developers      : Candidate Developer objects (project-specific pool).
        resolutions     : dict mapping bug.id -> developer.id (who fixed what).
        """
        self._historical_bugs = historical_bugs
        self._developers      = developers
        self._resolutions     = resolutions

        # Build per-developer bug-index and component maps.
        # Only project bugs in the resolutions dict are mapped — NLP context bugs
        # (Eclipse / Bugzilla) are not in resolutions so they are skipped.
        self._dev_bug_indices = {}
        self._dev_components  = {}
        for idx, bug in enumerate(historical_bugs):
            dev_id = resolutions.get(bug.id)
            if dev_id is None:
                continue
            if dev_id not in self._dev_bug_indices:
                self._dev_bug_indices[dev_id] = []
                self._dev_components[dev_id]  = set()
            self._dev_bug_indices[dev_id].append(idx)
            self._dev_components[dev_id].add(bug.module)

        # Fit TF-IDF on the full corpus; keep as sparse CSR matrix.
        corpus = [bug.description for bug in historical_bugs]
        self._historical_vectors = self._vectorizer.fit_transform(corpus)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def assign(self, new_bug: Bug, k: int = 20) -> tuple:
        """
        Assign a new bug to the most suitable developer.

        Parameters
        ----------
        new_bug : The Bug to assign.
        k       : Number of nearest historical neighbours to inspect (default 20).
                  Higher values widen the candidate pool, improving recall.

        Returns
        -------
        (best_developer, ranked_list, weights, signals)

        best_developer : Developer | None
        ranked_list    : list of (Developer, score) tuples, best first
        weights        : np.ndarray — 6-element weight vector unique to this bug
        signals        : dict — NLP signals that drove the weight calculation

        Returns (None, [], None, {}) if no assignment can be made.
        """
        if self._historical_vectors is None:
            raise RuntimeError("BugAssigner has not been trained yet. Call fit() first.")

        # Step 1 — Feature Extraction (sparse)
        new_vector = self._vectorizer.transform([new_bug.description])

        # Step 2 — KNN Similarity Search across all historical bugs
        # cosine_similarity handles sparse inputs natively; result is a dense 1-D array.
        similarities = cosine_similarity(new_vector, self._historical_vectors)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]

        # Step 3 — Candidate Extraction from the top-K similar bugs
        candidate_ids: set[int] = set()
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

        # Step 5 — KNN Affinity: per-developer topical expertise score.
        # For each candidate, compute the mean cosine similarity between the new bug
        # and every historical bug that developer previously fixed.
        # Developers who've fixed bugs similar to this one score higher here.
        knn_affinity: list[float] = []
        for d in available:
            indices = self._dev_bug_indices.get(d.id, [])
            if indices:
                aff = float(similarities[indices].mean())
            else:
                aff = 0.0
            knn_affinity.append(aff)

        # Step 6 — Attribute Matrix (6 columns per candidate)
        # [Experience, Fix Time, Success Rate, Workload, Domain Skill, KNN Affinity]
        matrix = np.array(
            [
                [d.experience, d.fix_time, d.success_rate, d.workload, d.domain_skill, knn_affinity[i]]
                for i, d in enumerate(available)
            ],
            dtype=float,
        )

        # Step 7 — Per-Bug Dynamic Weight Computation (unique for every bug)
        weights, signals = compute_dynamic_weights(new_bug)

        # Step 8a — Min-Max Normalisation + Weighted Sum Model (WSM)
        mins = matrix.min(axis=0)
        maxs = matrix.max(axis=0)
        norm = np.zeros_like(matrix)

        for j in range(matrix.shape[1]):
            rng = maxs[j] - mins[j]
            if rng == 0:
                norm[:, j] = 1.0   # all equal -> full score for everyone
            elif j in BENEFIT_COLS:
                norm[:, j] = (matrix[:, j] - mins[j]) / rng
            else:                  # cost column: invert so lower is better
                norm[:, j] = (maxs[j] - matrix[:, j]) / rng

        scores = norm @ weights

        # Step 8b — Component Match Bonus
        # A developer who has previously fixed bugs in the same module as this
        # bug gets a 10% score boost — they have demonstrated domain familiarity.
        for i, d in enumerate(available):
            if new_bug.module in self._dev_components.get(d.id, set()):
                scores[i] *= 1.1

        order  = scores.argsort()[::-1]
        ranked = [(available[i], float(scores[i])) for i in order]

        return ranked[0][0], ranked, weights, signals

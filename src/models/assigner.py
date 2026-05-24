"""
src/models/assigner.py

The core Bug-to-Developer Assignment algorithm.

Pipeline (8 steps):
  1. Feature Extraction     — Hybrid embeddings: dense (SentenceTransformers) + sparse (TF-IDF).
  2. Similarity Search      — Hybrid KNN cosine similarity against all historical bugs (K=20).
  3. Candidate Extraction   — Identify developers who fixed the top-K similar bugs.
  4. Hard Filtering         — Remove developers on leave or over 95% workload.
  5. KNN Affinity           — Per-developer topical expertise score: mean cosine similarity
                              between the new bug and every bug that developer has fixed.
  6. Attribute Matrix       — Build a matrix of 6 performance metrics per candidate.
  7. Semantic Prototype Weighting — Unique per-bug weights computed by measuring cosine
                              similarity of the bug description to 6 prototype sentences,
                              one per attribute dimension. No hardcoded keyword lists.
  8. WSM + Component Bonus  — Min-Max scale, weighted sum, apply component-match multiplier.

Accuracy Improvements (v4 — Semantic Prototype Weighting)
----------------------------------------------------------
Step 7 now uses the same SentenceTransformer model to drive weight computation.
Six prototype sentences define the "ideal bug" for each attribute dimension.
The bug's embedding is compared to each prototype via cosine similarity, and the
normalised similarity scores become the weight vector.

Advantages over keyword heuristics:
  • No hardcoded keyword lists to maintain.
  • Understands synonyms: "segfault" → urgency, even without "crash" in the text.
  • The weight tuning interface is plain English: rewrite a prototype sentence to
    change how that attribute is weighted.
  • Severity is still blended in as a multiplicative signal on the relevant prototypes.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.bug import Bug
from src.models.developer import Developer


# ---------------------------------------------------------------------------
# Semantic Prototype Sentences  (one per attribute column)
# ---------------------------------------------------------------------------
# To tune how an attribute is weighted, simply rewrite its prototype sentence
# in plain English. No code formulas need to change.
#
# Column order: [Experience, Fix Time, Success Rate, Workload, Domain Skill, KNN Affinity]

PROTOTYPE_SENTENCES = [
    # 0 — Experience
    (
        "This bug requires a senior engineer with deep historical knowledge of the codebase. "
        "It involves legacy systems, architectural decisions, or complex subsystems that only "
        "an experienced developer who has worked on this project for a long time can understand. "
        "Requires principal-level expertise, code ownership, and long-term project familiarity."
    ),
    # 1 — Fix Time  (high similarity → speed matters → lower fix time preferred)
    (
        "This is an emergency requiring an immediate hotfix. The system is down in production, "
        "customers are impacted, and the fix must be deployed as fast as possible. "
        "Speed is the top priority. Urgent outage, critical regression, blocker, showstopper, "
        "rapid turnaround needed, P0 or P1 severity, cannot wait."
    ),
    # 2 — Success Rate
    (
        "This is a high-stakes bug in mission-critical code with zero tolerance for errors. "
        "The fix must be correct the first time — a wrong fix could cause data loss or a "
        "security breach. Needs a developer with a strong track record of successful, "
        "reliable, and thoroughly tested resolutions. Quality over speed."
    ),
    # 3 — Workload  (high similarity → anyone free can handle it)
    (
        "This is a trivial, low-priority, non-blocking bug that any available developer can "
        "handle. It is a simple cosmetic fix, minor wording change, or a one-line tweak. "
        "Assign to whoever has the most free capacity. No special skill needed."
    ),
    # 4 — Domain Skill
    (
        "This bug is isolated to a very specific feature module or technical domain. "
        "It requires specialised knowledge of that particular component, API, or subsystem. "
        "Only a developer who is an expert in this specific area of the codebase can "
        "diagnose and resolve it efficiently."
    ),
    # 5 — KNN Affinity
    (
        "This bug closely resembles previously reported issues and follows a well-known "
        "pattern seen in past bug reports. Assign to the developer who has historically "
        "resolved the most similar bugs, as they already understand the root cause pattern "
        "and will have the relevant context from past fixes."
    ),
]

# Column indices in the 6-column attribute matrix
# [Experience, Fix Time, Success Rate, Workload, Domain Skill, KNN Affinity]
BENEFIT_COLS = {0, 2, 4, 5}   # Experience, Success Rate, Domain Skill, KNN Affinity
COST_COLS    = {1, 3}          # Fix Time, Workload

PROTOTYPE_NAMES = [
    "experience", "fix_time", "success_rate", "workload", "domain_skill", "knn_affinity"
]


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
        # Semantic embeddings using SentenceTransformers
        self._dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._sparse_model = TfidfVectorizer(
            stop_words="english",
            max_df=0.8,
            min_df=1,
            ngram_range=(1, 2)
        )
        self._historical_bugs: list[Bug] = []
        self._historical_dense = None            # dense numpy array
        self._historical_sparse = None           # sparse CSR matrix
        self._developers: list[Developer] = []
        self._resolutions: dict = {}             # bug_id -> developer_id

        # Per-developer lookup tables built at fit() time
        self._dev_bug_indices: dict[int, list[int]] = {}  # dev_id -> [hist indices]
        self._dev_components: dict[int, set[str]]   = {}  # dev_id -> {module, ...}
        self._bug_id_to_idx: dict[int, int]         = {}  # bug_id -> hist index

        # Pre-encode the 6 prototype sentences once at init time.
        # Shape: (6, embedding_dim) — reused for every assign() call.
        self._prototype_vectors = self._dense_model.encode(
            PROTOTYPE_SENTENCES, show_progress_bar=False
        )
        
        # LTR Specific additions
        self._ltr_model = None          # XGBRanker, loaded if available
        self._use_ltr = False           # A/B switch
        self._ltr_confidence_min = 0.1  # minimum score gap to trust LTR

    def load_ltr_model(self, path: str):
        """Load a trained LTR model. Call after fit()."""
        import xgboost as xgb
        self._ltr_model = xgb.XGBRanker()
        self._ltr_model.load_model(path)
        self._use_ltr = True
        print(f"[LTR] Model loaded from {path}")

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
        self._bug_id_to_idx   = {}
        for idx, bug in enumerate(historical_bugs):
            self._bug_id_to_idx[bug.id] = idx
            dev_id = resolutions.get(bug.id)
            if dev_id is None:
                continue
            if dev_id not in self._dev_bug_indices:
                self._dev_bug_indices[dev_id] = []
                self._dev_components[dev_id]  = set()
            self._dev_bug_indices[dev_id].append(idx)
            self._dev_components[dev_id].add(bug.module)

        # Encode corpus to dense embeddings and sparse TF-IDF vectors
        corpus = [bug.description for bug in historical_bugs]
        self._historical_dense = self._dense_model.encode(corpus, show_progress_bar=False)
        self._historical_sparse = self._sparse_model.fit_transform(corpus)

    # ------------------------------------------------------------------
    # LTR Utilities
    # ------------------------------------------------------------------
    
    def _get_candidate_features(self, bug: Bug, k: int = 20) -> list | None:
        """
        Run Steps 1-6 of the pipeline and return raw feature vectors
        per candidate developer. Used by LTR training and inference.

        Returns
        -------
        list of (Developer, feature_vector) tuples, or None if no candidates.
        """
        if self._historical_dense is None:
            raise RuntimeError("BugAssigner not trained. Call fit() first.")

        bug_idx = self._bug_id_to_idx.get(bug.id)
        if bug_idx is not None:
            new_dense = self._historical_dense[bug_idx:bug_idx+1]
            new_sparse = self._historical_sparse[bug_idx]
        else:
            new_dense = self._dense_model.encode([bug.description])
            new_sparse = self._sparse_model.transform([bug.description])

        # Step 2 — Hybrid Similarity
        sim_dense = cosine_similarity(new_dense, self._historical_dense)[0]
        sim_sparse = cosine_similarity(new_sparse, self._historical_sparse)[0]
        similarities = (sim_dense * 0.5) + (sim_sparse * 0.5)
        
        if bug_idx is not None:
            similarities[bug_idx] = -1.0  # Prevent self-retrieval
            
        top_k_indices = similarities.argsort()[-k:][::-1]

        # Step 3 — Candidate Extraction
        candidate_ids = set()
        for idx in top_k_indices:
            b = self._historical_bugs[idx]
            if b.id in self._resolutions:
                candidate_ids.add(self._resolutions[b.id])

        # Step 4 — Hard Filtering
        available = [
            d for d in self._developers
            if d.id in candidate_ids and not d.on_leave and d.workload < 0.95
        ]

        if not available:
            return None

        # Step 5 — KNN Affinity
        knn_affinity = []
        for d in available:
            indices = self._dev_bug_indices.get(d.id, [])
            if bug_idx is not None:
                indices = [idx_ for idx_ in indices if idx_ != bug_idx]
                
            if indices:
                knn_affinity.append(float(similarities[indices].mean()))
            else:
                knn_affinity.append(0.0)

        # Prototype scores (computed once, same for all candidates)
        text = bug.description + " " + bug.severity
        bug_vec = self._dense_model.encode([text])
        raw_sims = cosine_similarity(bug_vec, self._prototype_vectors)[0]
        prototype_scores = list((raw_sims + 1.0) / 2.0)  # map to [0, 1]

        # Build feature vectors
        result = []
        for i, dev in enumerate(available):
            dev_indices = self._dev_bug_indices.get(dev.id, [])
            if bug_idx is not None:
                dev_indices = [idx_ for idx_ in dev_indices if idx_ != bug_idx]

            features = [
                # Category A: Developer attributes (6)
                dev.experience,
                dev.fix_time,
                dev.success_rate,
                dev.workload,
                dev.domain_skill,
                knn_affinity[i],

                # Category B: Bug-Developer interaction (4)
                1.0 if bug.module in self._dev_components.get(dev.id, set()) else 0.0,
                float(len(dev_indices)),
                float(similarities[dev_indices].max()) if dev_indices else 0.0,
                float(similarities[dev_indices].mean()) if dev_indices else 0.0,

                # Category C: Bug context (3)
                1.0 if bug.severity == "critical" else 0.0,
                1.0 if bug.severity == "normal" else 0.0,
                1.0 if bug.severity == "minor" else 0.0,

                # Category D: Prototype scores (6)
                *prototype_scores,
            ]

            result.append((dev, features))

        return result
        
    def _assign_ltr(self, bug, available, similarities, k):
        """Score candidates using the trained LTR model with confidence fallback."""

        candidate_data = self._get_candidate_features(bug, k)
        if candidate_data is None:
            return None, [], None, {}

        devs = [d for d, _ in candidate_data]
        X = np.array([f for _, f in candidate_data], dtype=float)

        # LTR prediction
        scores = self._ltr_model.predict(X)

        # ---- CONFIDENCE CHECK (Safety Rule S3) ----
        sorted_scores = np.sort(scores)[::-1]
        score_gap = None
        if len(sorted_scores) >= 2:
            score_gap = sorted_scores[0] - sorted_scores[1]
            if score_gap < self._ltr_confidence_min:
                # Model is not confident → fall back to WSM
                return self._assign_wsm_fallback(bug, available, similarities)

        # Rank by LTR scores
        order = scores.argsort()[::-1]
        ranked = [(devs[i], float(scores[i])) for i in order]

        # Signals for debugging/display
        weights, signals = self._compute_semantic_weights(bug)
        signals["ranking_method"] = "ltr"
        signals["ltr_confidence_gap"] = round(float(score_gap), 4) if len(sorted_scores) >= 2 else None

        return ranked[0][0], ranked, weights, signals
        
    def _assign_wsm_fallback(self, bug, available, similarities):
        # We need the matrix to call WSM
        knn_affinity = []
        for d in available:
            indices = self._dev_bug_indices.get(d.id, [])
            if indices:
                knn_affinity.append(float(similarities[indices].mean()))
            else:
                knn_affinity.append(0.0)

        matrix = np.array(
            [
                [d.experience, d.fix_time, d.success_rate, d.workload, d.domain_skill, knn_affinity[i]]
                for i, d in enumerate(available)
            ],
            dtype=float,
        )
        return self._assign_wsm(bug, available, similarities, matrix, 20)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _assign_wsm(self, new_bug, available, similarities, matrix, k):
        # Step 7 — Semantic Prototype Weight Computation (unique for every bug)
        weights, signals = self._compute_semantic_weights(new_bug)

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
        if self._historical_dense is None or self._historical_sparse is None:
            raise RuntimeError("BugAssigner has not been trained yet. Call fit() first.")

        # Check for A/B switch
        use_ltr = self._use_ltr and os.environ.get("DISABLE_LTR") != "1"

        # Step 1 — Feature Extraction (dense semantic + sparse lexical)
        bug_idx = self._bug_id_to_idx.get(new_bug.id)
        if bug_idx is not None:
            new_dense = self._historical_dense[bug_idx:bug_idx+1]
            new_sparse = self._historical_sparse[bug_idx]
        else:
            new_dense = self._dense_model.encode([new_bug.description])
            new_sparse = self._sparse_model.transform([new_bug.description])

        # Step 2 — Hybrid Similarity Search
        # Combine dense semantic similarity with sparse lexical exact-matching
        sim_dense = cosine_similarity(new_dense, self._historical_dense)[0]
        sim_sparse = cosine_similarity(new_sparse, self._historical_sparse)[0]
        
        # Hybrid score weights both models equally
        similarities = (sim_dense * 0.5) + (sim_sparse * 0.5)
        
        if bug_idx is not None:
            similarities[bug_idx] = -1.0  # Prevent self-retrieval
            
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
            if bug_idx is not None:
                indices = [idx_ for idx_ in indices if idx_ != bug_idx]
                
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
        
        # Dual logging during development
        # wsm_result = self._assign_wsm(new_bug, available, similarities, matrix, k)
        # if use_ltr and self._ltr_model is not None:
        #    ltr_result = self._assign_ltr(new_bug, available, similarities, k)
        #    if wsm_result[0] and ltr_result[0] and wsm_result[0].id != ltr_result[0].id:
        #        print(f"[DISAGREE] Bug {new_bug.id}: WSM→{wsm_result[0].id}, LTR→{ltr_result[0].id}")

        if use_ltr and self._ltr_model is not None:
            # ---- LTR PATH ----
            return self._assign_ltr(new_bug, available, similarities, k)
        else:
            # ---- WSM PATH (existing code, untouched) ----
            return self._assign_wsm(new_bug, available, similarities, matrix, k)


    # ------------------------------------------------------------------
    # Semantic Prototype Weighting
    # ------------------------------------------------------------------

    def _compute_semantic_weights(self, bug: Bug) -> tuple:
        """
        Compute a unique 6-element weight vector by measuring cosine similarity
        between the bug description and each of the 6 prototype sentences.

        The prototype sentences act as "ideal bug descriptions" for each attribute.
        A bug that is semantically close to the Fix Time prototype (urgent, fast, P0)
        will receive a high weight on the Fix Time attribute — no keyword lists needed.

        Severity acts as a multiplicative booster on the most relevant prototypes:
          critical  → boosts Fix Time (1) and Success Rate (2)
          minor     → boosts Workload (3)  (assign to whoever is free)

        Returns
        -------
        (weights, signals)
        weights : np.ndarray — 6 elements, normalised to sum exactly to 1.0
        signals : dict       — similarity scores and severity for display / debugging
        """
        # Encode the bug using just the dense model (prototype comparison is semantic)
        text = bug.description + " " + bug.severity
        bug_vec = self._dense_model.encode([text])  # shape (1, dim)

        # Cosine similarity to each of the 6 prototypes → raw scores in [-1, 1]
        # Shift to [0, 1] by applying (score + 1) / 2 so negatives don't cause issues
        raw_sims = cosine_similarity(bug_vec, self._prototype_vectors)[0]  # shape (6,)
        raw_scores = (raw_sims + 1.0) / 2.0   # map [-1,1] -> [0,1]

        # Severity multiplier: amplifies the prototypes most relevant to severity
        severity_score = {"critical": 1.0, "normal": 0.5, "minor": 0.0}.get(bug.severity, 0.5)

        # critical bugs → boost Fix Time (1) and Success Rate (2)
        # minor bugs    → boost Workload (3) so they go to the freest person
        severity_boost = np.ones(6, dtype=float)
        severity_boost[1] *= 1.0 + 0.5 * severity_score          # Fix Time
        severity_boost[2] *= 1.0 + 0.3 * severity_score          # Success Rate
        severity_boost[3] *= 1.0 + 0.4 * (1.0 - severity_score)  # Workload (minor)

        boosted = raw_scores * severity_boost

        # Add a small floor so no attribute weight ever collapses to zero
        floored = boosted + 0.05

        # Normalise to sum exactly to 1.0
        weights = floored / floored.sum()

        signals = {
            name: round(float(raw_sims[i]), 4)
            for i, name in enumerate(PROTOTYPE_NAMES)
        }
        signals["severity_score"] = round(severity_score, 3)
        signals["weight_vector"] = [round(float(w), 4) for w in weights]

        return weights, signals

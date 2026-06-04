"""
src/models/ltr_data_builder.py

Builds training data for the Learning-to-Rank model by replaying the
assignment pipeline on historical bugs with known resolutions.

Feature Vector (19 dimensions)
-------------------------------
Category A — Developer static attributes (6):
    experience, fix_time, success_rate, workload, domain_skill, knn_affinity

Category B — Bug-developer interaction (4):
    component_match, num_past_bugs_fixed, cosine_sim_max, cosine_sim_mean

Category C — Bug severity flags (3):
    severity_is_critical, severity_is_normal, severity_is_minor

Category D — Semantic prototype similarity scores (6):
    proto_experience, proto_fix_time, proto_success_rate,
    proto_workload, proto_domain_skill, proto_knn_affinity
"""

from __future__ import annotations

import numpy as np
from src.models.bug import Bug
from src.models.developer import Developer


class LTRDataBuilder:
    """
    Replays the candidate extraction pipeline for each training bug
    and builds (features, labels, groups) suitable for XGBRanker.
    """

    def __init__(self, assigner) -> None:
        """
        Parameters
        ----------
        assigner : BugAssigner (already fit)
            The trained assigner whose pipeline we replay.
        """
        self.assigner = assigner

    def build_training_data(
        self,
        train_bugs: list,
        resolutions: dict,
        k: int = 20,
    ) -> tuple:
        """
        Replay the pipeline for each bug and collect feature rows.

        Parameters
        ----------
        train_bugs : list of Bug — training split only
        resolutions : dict — bug_id -> developer_id
        k : int — number of KNN neighbors (same as assign())

        Returns
        -------
        X : np.ndarray — shape (total_candidates, num_features)
        y : np.ndarray — shape (total_candidates,) — 1 for correct dev, 0 otherwise
        groups : list[int] — number of candidates per bug (for XGBRanker)
        feature_names : list[str] — column names for interpretability
        """
        all_features = []
        all_labels = []
        groups = []
        skipped = 0

        for bug in train_bugs:
            actual_dev_id = resolutions.get(bug.id)
            if actual_dev_id is None:
                skipped += 1
                continue

            # Get candidate features using the assigner's internal pipeline
            candidate_data = self.assigner._get_candidate_features(bug, k=k)

            if candidate_data is None or len(candidate_data) == 0:
                skipped += 1
                continue

            bug_features = []
            bug_labels = []

            for dev, features in candidate_data:
                bug_features.append(features)
                label = 1 if dev.id == actual_dev_id else 0
                bug_labels.append(label)

            # Only include this bug if the correct developer is in the candidate pool
            if sum(bug_labels) == 0:
                skipped += 1
                continue

            all_features.extend(bug_features)
            all_labels.extend(bug_labels)
            groups.append(len(bug_features))

        print(f"[LTR Data] Built {len(all_features)} rows from "
              f"{len(groups)} bugs (skipped {skipped})")

        X = np.array(all_features, dtype=float)
        y = np.array(all_labels, dtype=float)

        feature_names = self._get_feature_names()

        return X, y, groups, feature_names

    def _get_feature_names(self) -> list[str]:
        """Return the ordered feature column names matching the 19-dim feature vector."""
        return [
            # Category A: Developer static attributes (6)
            "experience",
            "fix_time",
            "success_rate",
            "workload",
            "domain_skill",
            "knn_affinity",
            # Category B: Bug-developer interaction (4)
            "component_match",
            "num_past_bugs_fixed",
            "cosine_sim_max",
            "cosine_sim_mean",
            # Category C: Bug severity one-hot flags (3)
            "severity_is_critical",
            "severity_is_normal",
            "severity_is_minor",
            # Category D: Semantic prototype similarity scores (6)
            "proto_experience",
            "proto_fix_time",
            "proto_success_rate",
            "proto_workload",
            "proto_domain_skill",
            "proto_knn_affinity",
        ]

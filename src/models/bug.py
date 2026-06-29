"""
src/models/bug.py
Defines the Bug data class used throughout the pipeline.
"""

from __future__ import annotations


class Bug:
    """
    Represents a single bug report.

    Attributes
    ----------
    id          : Unique identifier (int or str).
    description : Full text of the bug (title + body). Used for NLP similarity search.
    severity    : Normalised severity — one of 'critical', 'normal', or 'minor'.
                  Drives the semantic prototype weight selection.
    module      : The component/area this bug belongs to (e.g. 'ui', 'backend').
    vector      : TF-IDF feature vector populated by BugAssigner at inference time.
    """

    __slots__ = ("id", "description", "severity", "module", "vector", "created_at")

    def __init__(
        self,
        id: int | str,
        description: str,
        severity: str,
        module: str,
        created_at: float = 0.0,
    ) -> None:
        self.id: int | str = id
        self.description: str = description
        self.severity: str = severity.lower()
        self.module: str = module
        self.created_at: float = created_at
        self.vector = None  # Populated by BugAssigner at fit/inference time

    def __repr__(self) -> str:
        return f"Bug(id={self.id!r}, severity={self.severity!r}, module={self.module!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bug):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

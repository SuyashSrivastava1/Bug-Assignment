"""
src/models/bug.py
Defines the Bug data class used throughout the pipeline.
"""

class Bug:
    """
    Represents a single bug report.

    Attributes:
        id          : Unique identifier (int or str)
        description : The full text of the bug (title + body). Used for NLP similarity search.
        severity    : 'critical', 'normal', or 'minor'. Drives the AI weight selection.
        module      : The component/area this bug belongs to (e.g. 'ui', 'backend').
        vector      : TF-IDF feature vector set by BugAssigner at inference time.
    """

    def __init__(self, id, description: str, severity: str, module: str):
        self.id = id
        self.description = description
        self.severity = severity.lower()
        self.module = module
        self.vector = None   # Populated by BugAssigner.assign_bug()

    def __repr__(self):
        return f"Bug(id={self.id}, severity={self.severity}, module={self.module})"

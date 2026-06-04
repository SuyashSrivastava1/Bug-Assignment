"""
src/models/developer.py
Defines the Developer data class used throughout the pipeline.
"""

from __future__ import annotations


class Developer:
    """
    Represents a software developer with measurable performance metrics.

    All metrics are derived from historical bug data — no synthetic/random values.

    Attributes
    ----------
    id           : Unique integer identifier assigned at build time.
    name         : Display name or GitHub username.
    experience   : Total bugs historically fixed. Benefit — higher is better.
    fix_time     : Average hours to close a bug. Cost — lower is better.
    success_rate : Percentage of bugs closed without re-opening. Benefit — higher is better.
                   Defaults to 100 when re-open data is unavailable.
    workload     : Current open-issue load as a float 0–100. Cost — lower is better.
                   Defaults to 0 when live open-issue data is unavailable.
    domain_skill : Score reflecting breadth of component expertise.
                   Benefit — higher means broader coverage.
    on_leave     : If True the developer is hard-filtered from all candidates.
    """

    __slots__ = (
        "id", "name", "experience", "fix_time",
        "success_rate", "workload", "domain_skill", "on_leave",
    )

    def __init__(
        self,
        id: int,
        name: str,
        experience: int,
        fix_time: float,
        success_rate: float,
        workload: float,
        domain_skill: float,
        on_leave: bool = False,
    ) -> None:
        self.id: int = id
        self.name: str = name
        self.experience: int = experience
        self.fix_time: float = fix_time
        self.success_rate: float = success_rate
        self.workload: float = workload
        self.domain_skill: float = domain_skill
        self.on_leave: bool = on_leave

    def __repr__(self) -> str:
        return (
            f"Developer(id={self.id}, name={self.name!r}, "
            f"exp={self.experience}, fix_time={self.fix_time:.1f}h, "
            f"workload={self.workload:.0f}%)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Developer):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

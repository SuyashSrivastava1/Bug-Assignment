"""
src/models/developer.py
Defines the Developer data class used throughout the pipeline.
"""

class Developer:
    """
    Represents a software developer with measurable performance metrics.

    All metrics are derived purely from historical bug data — no dummy/random values.

    Attributes:
        id           : Unique integer identifier assigned at build time.
        name         : Display name or GitHub username.
        experience   : Total number of bugs historically fixed (Benefit — higher is better).
        fix_time     : Average hours taken to close a bug (Cost — lower is better).
        success_rate : Percentage of bugs closed without re-opening (Benefit — higher is better).
                       Set to 100 when re-open data is unavailable.
        workload     : Current open issue load as a percentage 0-100 (Cost — lower is better).
                       Set to 0 when live open-issue data is unavailable.
        domain_skill : Score reflecting how many distinct components the developer has worked on
                       (Benefit — higher means broader expertise).
        on_leave     : If True, developer is hard-filtered out of all candidates.
    """

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
    ):
        self.id = id
        self.name = name
        self.experience = experience
        self.fix_time = fix_time
        self.success_rate = success_rate
        self.workload = workload
        self.domain_skill = domain_skill
        self.on_leave = on_leave

    def __repr__(self):
        return (
            f"Developer(id={self.id}, name={self.name!r}, "
            f"exp={self.experience}, fix_time={self.fix_time:.1f}h, "
            f"workload={self.workload}%)"
        )

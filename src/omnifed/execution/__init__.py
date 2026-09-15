"""Launch backends: Ray vs Slurm.

Phase 1 rehouses Engine dispatch only. Single-rank Slurm still runs
``src.omnifed.slurm_worker``. TorchTitan is not wired here yet.
"""

from .shared import uses_torchtitan, validate_execution_mode

__all__ = ["uses_torchtitan", "validate_execution_mode"]

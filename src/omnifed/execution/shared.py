from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


SUPPORTED_EXECUTION_MODES = frozenset({"ray", "slurm"})
SUPPORTED_CLIENT_RUNTIMES = frozenset({"slurm", "torchtitan"})


def validate_execution_mode(execution_mode: str) -> str:
    """Return normalized ``engine.mode``. Ray vs Slurm only."""
    mode = str(execution_mode).lower()
    if mode not in SUPPORTED_EXECUTION_MODES:
        supported = ", ".join(sorted(SUPPORTED_EXECUTION_MODES))
        raise ValueError(
            f"engine.mode must be one of: {supported}, got {execution_mode!r}"
        )
    return mode


def uses_torchtitan(cfg: Any) -> bool:
    """True when yaml asks for multi-GPU-inside-one-client (TorchTitan)."""
    runtime = OmegaConf.select(cfg, "engine.client_runtime", default="slurm")
    if runtime is not None and str(runtime).lower() in ("torchtitan", "titan"):
        return True
    backend = OmegaConf.select(cfg, "backend.internal_backend", default=None)
    if backend is not None and str(backend).lower() == "torchtitan":
        return True
    return bool(OmegaConf.select(cfg, "torchtitan.subclusters.enabled", default=False))

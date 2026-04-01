"""Contracts for persistence backends (PostgreSQL or no-op)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


@dataclass
class StepRecord:
    run_id: str
    step: int
    started_at: datetime
    ended_at: datetime
    train_loss: float
    avg_reward: float
    avg_kl: float
    geometric_mean: float
    eligible_for_scoring: bool
    env_scores: Dict[str, float]
    env_completeness: Dict[str, float]
    exported_checkpoint: bool
    weights_updated_at: Optional[datetime]


@dataclass
class CheckpointRecord:
    run_id: str
    step: int
    storage_path: Path
    geometric_mean: float
    env_scores: Dict[str, float]
    weights_updated_at: Optional[datetime]
    manifest_version: int
    bytes_total: Optional[int]
    shard_files: Optional[list[str]]


class RunPersistencePort(Protocol):
    """Registry for runs, orchestrator steps, and checkpoint metadata."""

    def begin_run(
        self,
        run_id: str,
        *,
        config_snapshot: Dict[str, Any],
        model_dir: Path,
        artifacts_dir: Path,
        git_commit: Optional[str] = None,
    ) -> None:
        ...

    def record_step(self, record: StepRecord) -> None:
        ...

    def record_checkpoint(self, record: CheckpointRecord) -> None:
        ...

    def finish_run(self, run_id: str, status: str) -> None:
        """status: 'completed' | 'failed'"""

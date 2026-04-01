"""No-op persistence when PostgreSQL is disabled."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .protocols import CheckpointRecord, RunPersistencePort, StepRecord


class NullRunPersistence(RunPersistencePort):
    def begin_run(
        self,
        run_id: str,
        *,
        config_snapshot: Dict[str, Any],
        model_dir: Path,
        artifacts_dir: Path,
        git_commit: Optional[str] = None,
    ) -> None:
        del run_id, config_snapshot, model_dir, artifacts_dir, git_commit

    def record_step(self, record: StepRecord) -> None:
        del record

    def record_checkpoint(self, record: CheckpointRecord) -> None:
        del record

    def finish_run(self, run_id: str, status: str) -> None:
        del run_id, status

"""Contracts for metrics / telemetry backends."""

from __future__ import annotations

from typing import Protocol

from ..schemas import EvalMetrics, TrainMetrics


class MetricsRecorderPort(Protocol):
    def start(self) -> None:
        """Bind HTTP /metrics if applicable."""

    def record_step(self, run_id: str, train: TrainMetrics, eval_: EvalMetrics) -> None:
        ...

    def record_checkpoint_exported(self, run_id: str) -> None:
        ...

    def record_weights_timestamp(self, run_id: str, unix_ts: float | None) -> None:
        ...

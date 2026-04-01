from __future__ import annotations

from ..schemas import EvalMetrics, TrainMetrics
from .protocols import MetricsRecorderPort


class NullMetricsRecorder(MetricsRecorderPort):
    def start(self) -> None:
        pass

    def record_step(self, run_id: str, train: TrainMetrics, eval_: EvalMetrics) -> None:
        del run_id, train, eval_

    def record_checkpoint_exported(self, run_id: str) -> None:
        del run_id

    def record_weights_timestamp(self, run_id: str, unix_ts: float | None) -> None:
        del run_id, unix_ts

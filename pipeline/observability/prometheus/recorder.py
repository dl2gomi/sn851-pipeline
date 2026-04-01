"""Prometheus metrics for Grafana (in-process scrape target)."""

from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, start_http_server

from ...schemas import EvalMetrics, TrainMetrics
from ..protocols import MetricsRecorderPort

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "sn851_"


class PrometheusMetricsRecorder(MetricsRecorderPort):
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._started = False
        self._train_loss = Gauge(
            f"{_METRIC_PREFIX}train_loss",
            "Orchestrator-step training loss",
            ["run_id"],
        )
        self._gm = Gauge(
            f"{_METRIC_PREFIX}geometric_mean",
            "Geometric mean eval metric",
            ["run_id"],
        )
        self._eligible = Gauge(
            f"{_METRIC_PREFIX}eligible_for_scoring",
            "1 if eligible for scoring",
            ["run_id"],
        )
        self._step = Gauge(
            f"{_METRIC_PREFIX}orchestrator_step",
            "Current orchestrator step",
            ["run_id"],
        )
        self._reward = Gauge(
            f"{_METRIC_PREFIX}avg_reward",
            "Average reward in training mini-batch",
            ["run_id"],
        )
        self._kl = Gauge(
            f"{_METRIC_PREFIX}avg_kl",
            "Average KL estimate",
            ["run_id"],
        )
        self._env_score = Gauge(
            f"{_METRIC_PREFIX}env_score",
            "Per-environment score",
            ["run_id", "env"],
        )
        self._weights_ts = Gauge(
            f"{_METRIC_PREFIX}weights_updated_timestamp",
            "Unix time of last weight update",
            ["run_id"],
        )
        self._ckpt_exported = Counter(
            f"{_METRIC_PREFIX}checkpoint_exported_total",
            "Number of full checkpoint exports",
            ["run_id"],
        )

    def start(self) -> None:
        if self._started:
            return
        try:
            start_http_server(self._port, addr=self._host)
            logger.info("Prometheus metrics on http://%s:%s/metrics", self._host, self._port)
        except OSError:
            logger.exception("Failed to bind Prometheus metrics server; metrics disabled for this process")
            return
        self._started = True

    def record_step(self, run_id: str, train: TrainMetrics, eval_: EvalMetrics) -> None:
        self._train_loss.labels(run_id=run_id).set(train.loss)
        self._gm.labels(run_id=run_id).set(eval_.geometric_mean)
        self._eligible.labels(run_id=run_id).set(1.0 if eval_.eligible_for_scoring else 0.0)
        self._step.labels(run_id=run_id).set(float(train.step))
        self._reward.labels(run_id=run_id).set(train.avg_reward)
        self._kl.labels(run_id=run_id).set(train.avg_kl)
        for env, score in eval_.env_scores.items():
            self._env_score.labels(run_id=run_id, env=str(env)).set(float(score))

    def record_checkpoint_exported(self, run_id: str) -> None:
        self._ckpt_exported.labels(run_id=run_id).inc()

    def record_weights_timestamp(self, run_id: str, unix_ts: float | None) -> None:
        if unix_ts is None:
            return
        self._weights_ts.labels(run_id=run_id).set(unix_ts)

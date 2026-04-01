"""Construct metrics recorders from pipeline config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .noop import NullMetricsRecorder
from .protocols import MetricsRecorderPort

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger(__name__)


def build_metrics_recorder(cfg: "PipelineConfig") -> MetricsRecorderPort:
    if not cfg.prometheus.enabled:
        return NullMetricsRecorder()
    try:
        from .prometheus.recorder import PrometheusMetricsRecorder

        return PrometheusMetricsRecorder(cfg.prometheus.host, cfg.prometheus.port)
    except ImportError:
        logger.warning("prometheus_client is not installed; metrics disabled")
        return NullMetricsRecorder()

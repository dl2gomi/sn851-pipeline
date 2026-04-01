"""Construct persistence backends from pipeline config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .noop import NullRunPersistence
from .protocols import RunPersistencePort

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger(__name__)


def build_run_persistence(cfg: "PipelineConfig") -> RunPersistencePort:
    if not cfg.postgres.enabled or not (cfg.postgres.dsn or "").strip():
        return NullRunPersistence()
    try:
        from .postgres.repository import PostgresRunRepository

        return PostgresRunRepository(cfg.postgres.dsn.strip())
    except ImportError:
        logger.warning("psycopg is not installed; PostgreSQL persistence disabled")
        return NullRunPersistence()

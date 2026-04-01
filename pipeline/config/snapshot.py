"""Serialize pipeline config for Postgres / audit (JSON-safe)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .pipeline_config import PipelineConfig


def pipeline_config_snapshot(cfg: PipelineConfig) -> dict[str, Any]:
    raw = asdict(cfg)

    def _fix(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_fix(v) for v in obj]
        return obj

    return json.loads(json.dumps(_fix(raw), default=str))

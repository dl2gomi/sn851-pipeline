"""PostgreSQL connection helpers (psycopg v3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def load_schema_ddl() -> str:
    return _SCHEMA_PATH.read_text(encoding="utf-8")


def connect(dsn: str) -> Any:
    import psycopg

    return psycopg.connect(dsn)

"""PostgreSQL implementation of run / step / checkpoint registry."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from psycopg.types.json import Json

from ..protocols import CheckpointRecord, RunPersistencePort, StepRecord
from .connection import connect, load_schema_ddl

logger = logging.getLogger(__name__)


class PostgresRunRepository(RunPersistencePort):
    """Connects to an external Postgres; failures are logged and do not raise (training continues)."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._schema_ready = False
        self._disabled = False

    def _ensure_schema(self, conn: Any) -> None:
        if self._schema_ready:
            return
        ddl = load_schema_ddl()
        for block in ddl.split(";"):
            stmt = block.strip()
            if not stmt:
                continue
            with conn.cursor() as cur:
                cur.execute(stmt + ";")
        conn.commit()
        self._schema_ready = True

    def begin_run(
        self,
        run_id: str,
        *,
        config_snapshot: Dict[str, Any],
        model_dir: Any,
        artifacts_dir: Any,
        git_commit: Optional[str] = None,
    ) -> None:
        if self._disabled:
            return
        try:
            with connect(self._dsn) as conn:
                self._ensure_schema(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO runs (id, status, config_snapshot, model_dir, artifacts_dir, git_commit)
                        VALUES (%s::uuid, 'running', %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            config_snapshot = EXCLUDED.config_snapshot,
                            model_dir = EXCLUDED.model_dir,
                            artifacts_dir = EXCLUDED.artifacts_dir,
                            git_commit = COALESCE(EXCLUDED.git_commit, runs.git_commit),
                            status = 'running',
                            finished_at = NULL
                        """,
                        (
                            run_id,
                            Json(config_snapshot),
                            str(model_dir),
                            str(artifacts_dir),
                            git_commit,
                        ),
                    )
                conn.commit()
        except Exception:
            logger.exception("Postgres begin_run failed; disabling further Postgres writes for this process")
            self._disabled = True

    def record_step(self, record: StepRecord) -> None:
        if self._disabled:
            return
        try:
            with connect(self._dsn) as conn:
                self._ensure_schema(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO orchestrator_steps (
                            run_id, step, started_at, ended_at,
                            train_loss, avg_reward, avg_kl, geometric_mean,
                            eligible_for_scoring, env_scores, env_completeness, exported_checkpoint,
                            weights_updated_at
                        )
                        VALUES (
                            %s::uuid, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s
                        )
                        ON CONFLICT (run_id, step) DO UPDATE SET
                            ended_at = EXCLUDED.ended_at,
                            train_loss = EXCLUDED.train_loss,
                            avg_reward = EXCLUDED.avg_reward,
                            avg_kl = EXCLUDED.avg_kl,
                            geometric_mean = EXCLUDED.geometric_mean,
                            eligible_for_scoring = EXCLUDED.eligible_for_scoring,
                            env_scores = EXCLUDED.env_scores,
                            env_completeness = EXCLUDED.env_completeness,
                            exported_checkpoint = EXCLUDED.exported_checkpoint,
                            weights_updated_at = EXCLUDED.weights_updated_at
                        """,
                        (
                            record.run_id,
                            record.step,
                            record.started_at,
                            record.ended_at,
                            record.train_loss,
                            record.avg_reward,
                            record.avg_kl,
                            record.geometric_mean,
                            record.eligible_for_scoring,
                            Json(record.env_scores),
                            Json(record.env_completeness),
                            record.exported_checkpoint,
                            record.weights_updated_at,
                        ),
                    )
                conn.commit()
        except Exception:
            logger.exception("Postgres record_step failed (run_id=%s step=%s)", record.run_id, record.step)

    def record_checkpoint(self, record: CheckpointRecord) -> None:
        if self._disabled:
            return
        try:
            cp_id = str(uuid.uuid4())
            with connect(self._dsn) as conn:
                self._ensure_schema(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO checkpoints (
                            id, run_id, step, storage_path, geometric_mean, env_scores,
                            weights_updated_at, manifest_version, bytes_total, shard_files
                        )
                        VALUES (
                            %s::uuid, %s::uuid, %s, %s, %s, %s,
                            %s, %s, %s, %s
                        )
                        """,
                        (
                            cp_id,
                            record.run_id,
                            record.step,
                            str(record.storage_path.resolve()),
                            record.geometric_mean,
                            Json(record.env_scores),
                            record.weights_updated_at,
                            record.manifest_version,
                            record.bytes_total,
                            Json(record.shard_files) if record.shard_files is not None else None,
                        ),
                    )
                conn.commit()
        except Exception:
            logger.exception(
                "Postgres record_checkpoint failed (run_id=%s step=%s)", record.run_id, record.step
            )

    def finish_run(self, run_id: str, status: str) -> None:
        if self._disabled:
            return
        if status not in ("completed", "failed"):
            status = "failed"
        try:
            with connect(self._dsn) as conn:
                self._ensure_schema(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE runs SET status = %s, finished_at = NOW()
                        WHERE id = %s::uuid
                        """,
                        (status, run_id),
                    )
                conn.commit()
        except Exception:
            logger.exception("Postgres finish_run failed (run_id=%s)", run_id)

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...config.pipeline_config import CheckpointConfig
from ...schemas import EvalMetrics

if TYPE_CHECKING:
    from ...integration.runtime import SharedModelRuntime

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Outcome of an export attempt on a gated best step."""

    exported: bool
    checkpoint_path: Optional[Path] = None


class Exporter:
    def __init__(
        self,
        artifacts_dir: Path,
        env_regression_tolerance: float,
        checkpoint_cfg: Optional[CheckpointConfig] = None,
    ):
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.env_regression_tolerance = env_regression_tolerance
        self.checkpoint_cfg = checkpoint_cfg or CheckpointConfig()
        self.best_metric = -1.0
        self.best_env_scores: Dict[str, float] = {}

    def _has_hard_regression(self, eval_metrics: EvalMetrics) -> bool:
        if not self.best_env_scores:
            return False
        for env, score in eval_metrics.env_scores.items():
            prev = self.best_env_scores.get(env)
            if prev is None:
                continue
            if (prev - score) > self.env_regression_tolerance:
                return True
        return False

    def export_if_best(
        self,
        eval_metrics: EvalMetrics,
        train_step_loss: Optional[float] = None,
        *,
        run_id: Optional[str] = None,
        runtime: Optional["SharedModelRuntime"] = None,
    ) -> ExportResult:
        if not eval_metrics.eligible_for_scoring:
            return ExportResult(False)
        if self._has_hard_regression(eval_metrics):
            return ExportResult(False)
        if eval_metrics.geometric_mean <= self.best_metric:
            return ExportResult(False)

        self.best_metric = eval_metrics.geometric_mean
        self.best_env_scores = dict(eval_metrics.env_scores)

        output_path = self.artifacts_dir / f"checkpoint_step_{eval_metrics.step:04d}.json"
        payload = {
            "run_id": run_id,
            "step": eval_metrics.step,
            "geometric_mean": eval_metrics.geometric_mean,
            "env_scores": eval_metrics.env_scores,
            "env_completeness": eval_metrics.env_completeness,
            "eligible_for_scoring": eval_metrics.eligible_for_scoring,
            "train_step_loss": train_step_loss,
        }

        ckpt_path: Optional[Path] = None
        cfg = self.checkpoint_cfg
        if cfg.enabled and runtime is not None:
            try:
                from .checkpoint_fs import save_checkpoint_atomic

                checkpoints_root = self.artifacts_dir / cfg.subdir
                manifest: Dict[str, Any] = {
                    **payload,
                    "manifest_version": 1,
                }
                ckpt_path = save_checkpoint_atomic(
                    runtime,
                    checkpoints_root=checkpoints_root,
                    step=eval_metrics.step,
                    run_id=run_id or "",
                    manifest=manifest,
                    atomic=cfg.atomic,
                    max_shard_size=cfg.max_shard_size,
                    safe_serialization=cfg.safe_serialization,
                    keep_last_n=cfg.keep_last_n,
                )
                payload["checkpoint_dir"] = str(ckpt_path.resolve())
            except Exception:
                logger.exception("Full checkpoint save failed; continuing with JSON manifest only")

        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ExportResult(exported=True, checkpoint_path=ckpt_path)

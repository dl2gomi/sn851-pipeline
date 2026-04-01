import json
from pathlib import Path
from typing import Dict, Optional

from ...schemas import EvalMetrics


class Exporter:
    def __init__(self, artifacts_dir: Path, env_regression_tolerance: float):
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.env_regression_tolerance = env_regression_tolerance
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

    def export_if_best(self, eval_metrics: EvalMetrics, train_step_loss: Optional[float] = None) -> bool:
        if not eval_metrics.eligible_for_scoring:
            return False
        if self._has_hard_regression(eval_metrics):
            return False
        if eval_metrics.geometric_mean <= self.best_metric:
            return False

        self.best_metric = eval_metrics.geometric_mean
        self.best_env_scores = dict(eval_metrics.env_scores)
        output_path = self.artifacts_dir / f"checkpoint_step_{eval_metrics.step:04d}.json"
        payload = {
            "step": eval_metrics.step,
            "geometric_mean": eval_metrics.geometric_mean,
            "env_scores": eval_metrics.env_scores,
            "env_completeness": eval_metrics.env_completeness,
            "eligible_for_scoring": eval_metrics.eligible_for_scoring,
            "train_step_loss": train_step_loss,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True

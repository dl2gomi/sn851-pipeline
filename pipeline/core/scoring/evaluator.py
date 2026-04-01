from math import prod
from statistics import mean
from typing import Dict, List, Set

from ...schemas import EnvConfig, EvalMetrics, Trajectory


class Evaluator:
    def __init__(self, env_configs: Dict[str, EnvConfig], geometric_mean_epsilon: float):
        self.env_configs = env_configs
        self.epsilon = geometric_mean_epsilon

    def evaluate(
        self,
        step: int,
        trajectories: List[Trajectory],
        completed_task_ids: Dict[str, Set[int]],
    ) -> EvalMetrics:
        env_to_scores: Dict[str, List[float]] = {}
        for t in trajectories:
            env_to_scores.setdefault(t.task.env, []).append(t.raw_score)

        scoring_envs = [
            env_name for env_name, cfg in self.env_configs.items() if cfg.enabled_for_scoring
        ]
        env_scores: Dict[str, float] = {}
        env_completeness: Dict[str, float] = {}
        env_valid_for_scoring: Dict[str, bool] = {}
        for env in scoring_envs:
            scores = env_to_scores.get(env, [])
            env_scores[env] = mean(scores) if scores else 0.0

            sampling_list = self.env_configs[env].sampling_list
            expected_count = len(sampling_list)
            completed = len(completed_task_ids.get(env, set()) & set(sampling_list))
            completeness = (completed / expected_count) if expected_count > 0 else 0.0
            env_completeness[env] = completeness
            env_valid_for_scoring[env] = completeness >= self.env_configs[env].min_completeness

        if env_scores:
            vals = [v + self.epsilon for v in env_scores.values()]
            geometric_mean = max((prod(vals) ** (1.0 / len(vals))) - self.epsilon, 0.0)
        else:
            geometric_mean = 0.0

        eligible = bool(env_scores) and all(env_valid_for_scoring.values())
        return EvalMetrics(
            step=step,
            env_scores=env_scores,
            env_completeness=env_completeness,
            env_valid_for_scoring=env_valid_for_scoring,
            geometric_mean=geometric_mean,
            eligible_for_scoring=eligible,
        )

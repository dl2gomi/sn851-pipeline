from collections import defaultdict
import math
import random
import time
from typing import Dict, List

from ...schemas import EnvConfig, Task


class TaskSampler:
    """Sampling-list based sampler aligned with Affine scheduling semantics."""

    def __init__(
        self,
        env_configs: Dict[str, EnvConfig],
        env_weights: Dict[str, float],
        allowed_envs: List[str] | None = None,
        *,
        enable_rate_limit: bool = True,
        rate_margin: float = 1.2,
        min_completion_hours: int = 48,
        rate_window_seconds: int = 3600,
    ):
        self.env_configs = env_configs
        self.env_weights = dict(env_weights)
        self.allowed_envs = set(allowed_envs or env_configs.keys())
        self.enable_rate_limit = enable_rate_limit
        self.rate_margin = rate_margin
        self.min_completion_hours = min_completion_hours
        self.rate_window_seconds = rate_window_seconds
        self._tail_cursors: Dict[str, int] = {}
        self._sample_counts = defaultdict(int)
        self._allocation_timestamps = defaultdict(list)
        for env, cfg in env_configs.items():
            self._tail_cursors[env] = max(len(cfg.sampling_list) - 1, 0)

    def sample(self, n: int) -> List[Task]:
        envs = [
            e
            for e, cfg in self.env_configs.items()
            if (
                cfg.enabled_for_sampling
                and len(cfg.sampling_list) > 0
                and e in self.allowed_envs
                and not self._should_skip_env(e)
            )
        ]
        if not envs:
            return []

        tasks: List[Task] = []
        weights = [
            self.env_weights.get(e, self.env_configs[e].scheduling_weight) for e in envs
        ]
        for _ in range(n):
            env = random.choices(envs, weights=weights, k=1)[0]
            tasks.append(self._next_task(env))
        return tasks

    def _next_task(self, env: str) -> Task:
        cfg = self.env_configs[env]
        idx = self._tail_cursors[env]
        task_id = cfg.sampling_list[idx]
        idx -= 1
        if idx < 0:
            idx = len(cfg.sampling_list) - 1
        self._tail_cursors[env] = idx
        self._sample_counts[env] += 1
        self._record_allocation(env)
        return Task(
            env=env,
            task_id=task_id,
            prompt=f"Solve affine task_id={task_id} for env={env}",
            metadata={"sampling_weight": self.env_weights.get(env, 1.0)},
        )

    def sample_counts(self) -> Dict[str, int]:
        return dict(self._sample_counts)

    def _should_skip_env(self, env: str) -> bool:
        if not self.enable_rate_limit:
            return False
        cfg = self.env_configs[env]
        allowed_per_hour = self._allowed_per_hour(cfg)
        if allowed_per_hour <= 0:
            return False
        now = time.time()
        self._cleanup_allocations(env, now)
        return len(self._allocation_timestamps[env]) >= allowed_per_hour

    def _allowed_per_hour(self, cfg: EnvConfig) -> int:
        rotation_rate = 0.0
        if cfg.rotation_interval > 0 and cfg.rotation_count > 0:
            rotation_rate = cfg.rotation_count * (3600.0 / cfg.rotation_interval) * self.rate_margin
        min_rate = 0.0
        if cfg.sampling_count > 0 and self.min_completion_hours > 0:
            min_rate = cfg.sampling_count / float(self.min_completion_hours)
        base = max(rotation_rate, min_rate)
        return int(math.ceil(base)) if base > 0 else 0

    def _cleanup_allocations(self, env: str, now: float) -> None:
        window_start = now - self.rate_window_seconds
        arr = self._allocation_timestamps[env]
        while arr and arr[0] < window_start:
            arr.pop(0)

    def _record_allocation(self, env: str) -> None:
        self._allocation_timestamps[env].append(time.time())

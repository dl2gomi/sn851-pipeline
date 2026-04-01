from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    env: str
    task_id: int
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task: Task
    response: str
    raw_score: float
    kl_estimate: float = 0.0
    is_timeout: bool = False
    is_format_valid: bool = True
    reward: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainMetrics:
    step: int
    loss: float
    avg_reward: float
    avg_raw_score: float
    avg_kl: float
    by_env_avg_reward: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    step: int
    env_scores: Dict[str, float]
    env_completeness: Dict[str, float]
    env_valid_for_scoring: Dict[str, bool]
    geometric_mean: float
    eligible_for_scoring: bool


@dataclass
class EnvConfig:
    env_name: str
    enabled_for_sampling: bool
    enabled_for_scoring: bool
    min_completeness: float
    scheduling_weight: float
    sampling_list: List[int]
    sampling_count: int = 0
    rotation_enabled: bool = True
    rotation_count: int = 0
    rotation_interval: int = 3600
    display_name: Optional[str] = None

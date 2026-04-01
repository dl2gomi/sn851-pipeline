from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class TrainingConfig:
    algorithm: str = "ppo"
    lr: float = 1e-5
    rollout_batch_size: int = 16
    mini_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 2
    clip_range: float = 0.2
    target_kl: float = 0.03
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0
    use_value_head: bool = True
    value_loss_coef: float = 0.5


@dataclass
class ScoringConfig:
    geometric_mean_epsilon: float = 0.1
    env_regression_tolerance: float = 0.01


@dataclass
class SamplingConfig:
    enable_rate_limit: bool = True
    rate_margin: float = 1.2
    min_completion_hours: int = 48
    rate_window_seconds: int = 3600


@dataclass
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = "local-dev"


@dataclass
class PostgresConfig:
    """External PostgreSQL process; pipeline connects via DSN only."""

    enabled: bool = False
    dsn: str = ""


@dataclass
class PrometheusConfig:
    """In-process /metrics HTTP for Grafana scrape targets."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 9100


@dataclass
class CheckpointConfig:
    """Full HF-style weight export on gated best steps (local disk only)."""

    enabled: bool = True
    subdir: str = "checkpoints"
    keep_last_n: int = 3
    atomic: bool = True
    safe_serialization: bool = True
    max_shard_size: str = "10GB"


@dataclass
class PipelineConfig:
    system_config_url: str = (
        "https://raw.githubusercontent.com/AffineFoundation/affine-cortex/"
        "refs/heads/main/affine/database/system_config.json"
    )
    env_sampling_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "GAME": 1.0,
            "NAVWORLD": 1.0,
            "LIVEWEB": 1.0,
            "SWE-INFINITE": 1.0,
        }
    )
    rollouts_per_step: int = 16
    replay_capacity: int = 4000
    eval_window_size: int = 256
    max_steps: int = 20
    export_every_n_steps: int = 5
    artifacts_dir: Path = Path("artifacts")
    model_dir: Path = Path("model")
    run_id: Optional[str] = None
    train_scoring_envs_only: bool = True
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

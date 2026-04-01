from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from .pipeline_config import PipelineConfig


@dataclass
class RunCommandConfig:
    steps: int = 5
    dry_run: bool = False


@dataclass
class PullModelCommandConfig:
    model_dir: Path = Path("model")


@dataclass
class CliConfig:
    run: RunCommandConfig = field(default_factory=RunCommandConfig)
    pull_model: PullModelCommandConfig = field(default_factory=PullModelCommandConfig)


@dataclass
class AppConfig:
    pipeline: PipelineConfig
    cli: CliConfig = field(default_factory=CliConfig)


def load_app_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pipeline_raw = raw.get("pipeline", {})
    cli_raw = raw.get("cli", {})

    cfg = PipelineConfig()
    cfg.system_config_url = str(pipeline_raw.get("system_config_url", cfg.system_config_url))
    cfg.rollouts_per_step = int(pipeline_raw.get("rollouts_per_step", cfg.rollouts_per_step))
    cfg.replay_capacity = int(pipeline_raw.get("replay_capacity", cfg.replay_capacity))
    cfg.eval_window_size = int(pipeline_raw.get("eval_window_size", cfg.eval_window_size))
    cfg.max_steps = int(pipeline_raw.get("max_steps", cfg.max_steps))
    cfg.export_every_n_steps = int(pipeline_raw.get("export_every_n_steps", cfg.export_every_n_steps))
    cfg.artifacts_dir = Path(pipeline_raw.get("artifacts_dir", str(cfg.artifacts_dir)))
    cfg.model_dir = Path(pipeline_raw.get("model_dir", str(cfg.model_dir)))
    if pipeline_raw.get("run_id") is not None:
        cfg.run_id = str(pipeline_raw["run_id"]).strip() or None
    cfg.train_scoring_envs_only = bool(
        pipeline_raw.get("train_scoring_envs_only", cfg.train_scoring_envs_only)
    )

    if "env_sampling_weights" in pipeline_raw and isinstance(pipeline_raw["env_sampling_weights"], dict):
        cfg.env_sampling_weights = {
            str(k): float(v) for k, v in pipeline_raw["env_sampling_weights"].items()
        }

    training_raw: Dict[str, Any] = pipeline_raw.get("training", {})
    cfg.training.algorithm = str(training_raw.get("algorithm", cfg.training.algorithm))
    cfg.training.lr = float(training_raw.get("lr", cfg.training.lr))
    cfg.training.rollout_batch_size = int(
        training_raw.get("rollout_batch_size", cfg.training.rollout_batch_size)
    )
    cfg.training.mini_batch_size = int(
        training_raw.get("mini_batch_size", training_raw.get("batch_size", cfg.training.mini_batch_size))
    )
    cfg.training.gradient_accumulation_steps = int(
        training_raw.get("gradient_accumulation_steps", cfg.training.gradient_accumulation_steps)
    )
    cfg.training.ppo_epochs = int(training_raw.get("ppo_epochs", cfg.training.ppo_epochs))
    cfg.training.clip_range = float(training_raw.get("clip_range", cfg.training.clip_range))
    cfg.training.target_kl = float(training_raw.get("target_kl", cfg.training.target_kl))
    cfg.training.kl_coef = float(training_raw.get("kl_coef", cfg.training.kl_coef))
    cfg.training.max_grad_norm = float(training_raw.get("max_grad_norm", cfg.training.max_grad_norm))
    cfg.training.use_value_head = bool(training_raw.get("use_value_head", cfg.training.use_value_head))
    cfg.training.value_loss_coef = float(
        training_raw.get("value_loss_coef", cfg.training.value_loss_coef)
    )

    scoring_raw: Dict[str, Any] = pipeline_raw.get("scoring", {})
    cfg.scoring.geometric_mean_epsilon = float(
        scoring_raw.get("geometric_mean_epsilon", cfg.scoring.geometric_mean_epsilon)
    )
    cfg.scoring.env_regression_tolerance = float(
        scoring_raw.get("env_regression_tolerance", cfg.scoring.env_regression_tolerance)
    )

    sampling_raw: Dict[str, Any] = pipeline_raw.get("sampling", {})
    cfg.sampling.enable_rate_limit = bool(
        sampling_raw.get("enable_rate_limit", cfg.sampling.enable_rate_limit)
    )
    cfg.sampling.rate_margin = float(sampling_raw.get("rate_margin", cfg.sampling.rate_margin))
    cfg.sampling.min_completion_hours = int(
        sampling_raw.get("min_completion_hours", cfg.sampling.min_completion_hours)
    )
    cfg.sampling.rate_window_seconds = int(
        sampling_raw.get("rate_window_seconds", cfg.sampling.rate_window_seconds)
    )

    serve_raw: Dict[str, Any] = pipeline_raw.get("serve", {})
    cfg.serve.host = str(serve_raw.get("host", cfg.serve.host))
    cfg.serve.port = int(serve_raw.get("port", cfg.serve.port))
    cfg.serve.api_key = str(serve_raw.get("api_key", cfg.serve.api_key))

    pg_raw: Dict[str, Any] = pipeline_raw.get("postgres", {})
    cfg.postgres.enabled = bool(pg_raw.get("enabled", cfg.postgres.enabled))
    cfg.postgres.dsn = str(pg_raw.get("dsn", cfg.postgres.dsn))

    prom_raw: Dict[str, Any] = pipeline_raw.get("prometheus", {})
    cfg.prometheus.enabled = bool(prom_raw.get("enabled", cfg.prometheus.enabled))
    cfg.prometheus.host = str(prom_raw.get("host", cfg.prometheus.host))
    cfg.prometheus.port = int(prom_raw.get("port", cfg.prometheus.port))

    ck_raw: Dict[str, Any] = pipeline_raw.get("checkpoint", {})
    cfg.checkpoint.enabled = bool(ck_raw.get("enabled", cfg.checkpoint.enabled))
    cfg.checkpoint.subdir = str(ck_raw.get("subdir", cfg.checkpoint.subdir))
    cfg.checkpoint.keep_last_n = int(ck_raw.get("keep_last_n", cfg.checkpoint.keep_last_n))
    cfg.checkpoint.atomic = bool(ck_raw.get("atomic", cfg.checkpoint.atomic))
    cfg.checkpoint.safe_serialization = bool(ck_raw.get("safe_serialization", cfg.checkpoint.safe_serialization))
    cfg.checkpoint.max_shard_size = str(ck_raw.get("max_shard_size", cfg.checkpoint.max_shard_size))

    run_raw = cli_raw.get("run", {})
    pull_raw = cli_raw.get("pull_model", {})
    cli_cfg = CliConfig(
        run=RunCommandConfig(
            steps=int(run_raw.get("steps", 5)),
            dry_run=bool(run_raw.get("dry_run", False)),
        ),
        pull_model=PullModelCommandConfig(
            model_dir=Path(pull_raw.get("model_dir", str(cfg.model_dir)))
        ),
    )
    return AppConfig(pipeline=cfg, cli=cli_cfg)

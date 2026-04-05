import asyncio
import subprocess
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from ..config import PipelineConfig, pipeline_config_snapshot
from ..core.scoring.evaluator import Evaluator
from ..core.scoring.exporter import ExportResult, Exporter
from ..core.scoring.reward import RewardService
from ..core.execution.rollout import RolloutWorker
from ..core.sampling.sampler import TaskSampler
from ..core.state.replay_buffer import ReplayBuffer
from ..core.training.trainer import Trainer
from ..integration.affine_config import load_env_configs, split_active_envs
from ..integration.backends import (
    AffineEnvironmentExecutor,
    AffinePolicyAdapter,
    AffineTrainerBackend,
    SharedModelRuntime,
)
from ..integration.model_guard import validate_local_config
from ..observability import build_metrics_recorder
from ..observability.protocols import MetricsRecorderPort
from ..persistence import build_run_persistence
from ..persistence.protocols import CheckpointRecord, RunPersistencePort, StepRecord
from ..schemas import EvalMetrics, TrainMetrics, Trajectory


def _git_head() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


class PipelineOrchestrator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        if not (cfg.run_id and str(cfg.run_id).strip()):
            cfg.run_id = str(uuid.uuid4())
        else:
            cfg.run_id = str(cfg.run_id).strip()
        self.run_id = cfg.run_id
        self.env_configs = load_env_configs(
            system_config_url=cfg.system_config_url,
        )
        sampling_envs, scoring_envs = split_active_envs(self.env_configs)
        if not scoring_envs:
            raise ValueError("No enabled_for_scoring environments found in system_config.")
        if not sampling_envs:
            raise ValueError("No enabled_for_sampling environments found in system_config.")

        allowed_envs = scoring_envs if cfg.train_scoring_envs_only else sampling_envs
        self.sampler = TaskSampler(
            self.env_configs,
            cfg.env_sampling_weights,
            allowed_envs=allowed_envs,
            enable_rate_limit=cfg.sampling.enable_rate_limit,
            rate_margin=cfg.sampling.rate_margin,
            min_completion_hours=cfg.sampling.min_completion_hours,
            rate_window_seconds=cfg.sampling.rate_window_seconds,
        )
        self.runtime = SharedModelRuntime(
            model_dir=cfg.model_dir,
            training_defaults=cfg.training,
            serve_host=cfg.serve.host,
            serve_port=cfg.serve.port,
            serve_api_key=cfg.serve.api_key,
        )
        self.rollout = RolloutWorker(
            policy=AffinePolicyAdapter(self.runtime),
            executor=AffineEnvironmentExecutor(runtime=self.runtime),
        )
        self.reward = RewardService()
        self.replay = ReplayBuffer(capacity=cfg.replay_capacity)
        self.trainer = Trainer(
            lr=cfg.training.lr,
            batch_size=cfg.training.rollout_batch_size,
            ppo_epochs=cfg.training.ppo_epochs,
            training_config=cfg.training,
            backend=AffineTrainerBackend(self.runtime),
        )
        self.evaluator = Evaluator(
            env_configs=self.env_configs,
            geometric_mean_epsilon=cfg.scoring.geometric_mean_epsilon,
        )
        self.exporter = Exporter(
            artifacts_dir=cfg.artifacts_dir,
            env_regression_tolerance=cfg.scoring.env_regression_tolerance,
            checkpoint_cfg=cfg.checkpoint,
        )
        self.persistence: RunPersistencePort = build_run_persistence(cfg)
        self.metrics: MetricsRecorderPort = build_metrics_recorder(cfg)

    def run(self, steps: int, dry_run: bool = True) -> None:
        config_path = self.cfg.model_dir / "config.json"
        if not config_path.exists():
            msg = f"{config_path} missing. Pass --model-dir that contains config.json."
            if dry_run:
                print(f"[warning] model check: {msg}")
            else:
                raise ValueError(msg)
        else:
            ok, reason = validate_local_config(config_path)
            if not ok:
                if dry_run:
                    print(f"[warning] model check: {reason}")
                else:
                    raise ValueError(reason)

        print(
            f"Starting pipeline: run_id={self.run_id}, steps={steps}, dry_run={dry_run}, "
            f"system_config_url={self.cfg.system_config_url}"
        )
        if dry_run:
            print(
                "Dry-run preflight passed. Wire real shared model runtime, policy/trainer adapters, "
                "and environment executor to run training."
            )
            return

        self.metrics.start()
        self.persistence.begin_run(
            self.run_id,
            config_snapshot=pipeline_config_snapshot(self.cfg),
            model_dir=self.cfg.model_dir,
            artifacts_dir=self.cfg.artifacts_dir,
            git_commit=_git_head(),
        )

        run_ok = False
        try:
            self.runtime.ensure_server_started()
            for step in range(1, steps + 1):
                t0 = datetime.now(timezone.utc)
                trajectories = self._collect_rollouts()
                self.replay.add_many(trajectories)
                train_metrics = self._train_step(step)
                eval_metrics = self._eval_step(step)
                export_result = self._maybe_export(step, eval_metrics, train_metrics.loss)
                t1 = datetime.now(timezone.utc)
                self._persist_step(
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                    export_result=export_result,
                    started_at=t0,
                    ended_at=t1,
                )
                self._log_step(train_metrics, eval_metrics, export_result.exported)
            run_ok = True
        finally:
            close = getattr(self.rollout.executor, "close", None)
            if callable(close):
                asyncio.run(close())
            self.runtime.stop_server()
            self.persistence.finish_run(self.run_id, "completed" if run_ok else "failed")

    def _persist_step(
        self,
        *,
        train_metrics: TrainMetrics,
        eval_metrics: EvalMetrics,
        export_result: ExportResult,
        started_at: datetime,
        ended_at: datetime,
    ) -> None:
        self.metrics.record_step(self.run_id, train_metrics, eval_metrics)
        wu = self.runtime.weights_updated_at
        self.metrics.record_weights_timestamp(self.run_id, wu)
        wdt = datetime.fromtimestamp(wu, tz=timezone.utc) if wu is not None else None

        self.persistence.record_step(
            StepRecord(
                run_id=self.run_id,
                step=train_metrics.step,
                started_at=started_at,
                ended_at=ended_at,
                train_loss=train_metrics.loss,
                avg_reward=train_metrics.avg_reward,
                avg_kl=train_metrics.avg_kl,
                geometric_mean=eval_metrics.geometric_mean,
                eligible_for_scoring=eval_metrics.eligible_for_scoring,
                env_scores=dict(eval_metrics.env_scores),
                env_completeness=dict(eval_metrics.env_completeness),
                exported_checkpoint=export_result.checkpoint_path is not None,
                weights_updated_at=wdt,
            )
        )

        if export_result.checkpoint_path is not None:
            from ..core.scoring.checkpoint_fs import directory_size_bytes, list_weight_artifacts
            p = export_result.checkpoint_path
            self.persistence.record_checkpoint(
                CheckpointRecord(
                    run_id=self.run_id,
                    step=eval_metrics.step,
                    storage_path=p,
                    geometric_mean=eval_metrics.geometric_mean,
                    env_scores=dict(eval_metrics.env_scores),
                    weights_updated_at=wdt,  # same runtime snapshot as step row
                    manifest_version=1,
                    bytes_total=directory_size_bytes(p),
                    shard_files=list_weight_artifacts(p),
                )
            )
            self.metrics.record_checkpoint_exported(self.run_id)

    def _collect_rollouts(self) -> List[Trajectory]:
        tasks = self.sampler.sample(self.cfg.training.rollout_batch_size)
        trajectories = self.rollout.rollout(tasks)
        return [self.reward.score_trajectory(t) for t in trajectories]

    def _train_step(self, step: int) -> TrainMetrics:
        # PPO: train on the same on-policy window we just collected (FIFO tail), not mini_batch_size.
        batch = self.replay.latest(self.cfg.training.rollout_batch_size)
        return self.trainer.update(step, batch)

    def _eval_step(self, step: int) -> EvalMetrics:
        window = self.replay.latest(self.cfg.eval_window_size)
        return self.evaluator.evaluate(step, window, self.replay.completed_task_ids())

    def _maybe_export(self, step: int, eval_metrics: EvalMetrics, train_loss: float) -> ExportResult:
        if step % self.cfg.export_every_n_steps != 0:
            return ExportResult(False)
        return self.exporter.export_if_best(
            eval_metrics,
            train_step_loss=train_loss,
            run_id=self.run_id,
            runtime=self.runtime,
        )

    def _log_step(self, train_metrics: TrainMetrics, eval_metrics: EvalMetrics, exported: bool) -> None:
        env_view = ", ".join(
            f"{env}:{score:.3f}" for env, score in sorted(eval_metrics.env_scores.items())
        )
        comp_view = ", ".join(
            f"{env}:{c:.2%}" for env, c in sorted(eval_metrics.env_completeness.items())
        )
        print(
            f"[run_id={self.run_id} step={train_metrics.step}] "
            f"loss={train_metrics.loss:.4f} "
            f"avg_reward={train_metrics.avg_reward:.4f} "
            f"avg_kl={train_metrics.avg_kl:.4f} "
            f"gm={eval_metrics.geometric_mean:.4f} "
            f"eligible={eval_metrics.eligible_for_scoring} "
            f"exported={exported} "
            f"envs=[{env_view}] "
            f"completeness=[{comp_view}]"
        )

from typing import List

from ..config import PipelineConfig
from ..integration.affine_config import load_env_configs, split_active_envs
from ..integration.backends import (
    AffineEnvironmentExecutor,
    AffinePolicyAdapter,
    AffineTrainerBackend,
    SharedModelRuntime,
)
from ..integration.model_guard import validate_local_config
from ..core.scoring.evaluator import Evaluator
from ..core.scoring.exporter import Exporter
from ..core.scoring.reward import RewardService
from ..core.execution.rollout import RolloutWorker
from ..core.sampling.sampler import TaskSampler
from ..core.state.replay_buffer import ReplayBuffer
from ..core.training.trainer import Trainer
from ..schemas import EvalMetrics, TrainMetrics, Trajectory


class PipelineOrchestrator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
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
            batch_size=cfg.training.mini_batch_size,
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
        )

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
            f"Starting pipeline: steps={steps}, dry_run={dry_run}, "
            f"system_config_url={self.cfg.system_config_url}"
        )
        if dry_run:
            print(
                "Dry-run preflight passed. Wire real shared model runtime, policy/trainer adapters, "
                "and environment executor to run training."
            )
            return
        self.runtime.ensure_server_started()
        try:
            for step in range(1, steps + 1):
                trajectories = self._collect_rollouts()
                self.replay.add_many(trajectories)
                train_metrics = self._train_step(step)
                eval_metrics = self._eval_step(step)
                exported = self._maybe_export(step, eval_metrics, train_metrics.loss)
                self._log_step(train_metrics, eval_metrics, exported)
        finally:
            # Best-effort resource cleanup.
            close = getattr(self.rollout.executor, "close", None)
            if callable(close):
                import asyncio
                asyncio.run(close())
            self.runtime.stop_server()

    def _collect_rollouts(self) -> List[Trajectory]:
        tasks = self.sampler.sample(self.cfg.training.rollout_batch_size)
        trajectories = self.rollout.rollout(tasks)
        return [self.reward.score_trajectory(t) for t in trajectories]

    def _train_step(self, step: int) -> TrainMetrics:
        batch = self.replay.latest(self.cfg.training.mini_batch_size)
        return self.trainer.update(step, batch)

    def _eval_step(self, step: int) -> EvalMetrics:
        window = self.replay.latest(self.cfg.eval_window_size)
        return self.evaluator.evaluate(step, window, self.replay.completed_task_ids())

    def _maybe_export(self, step: int, eval_metrics: EvalMetrics, train_loss: float) -> bool:
        if step % self.cfg.export_every_n_steps != 0:
            return False
        return self.exporter.export_if_best(eval_metrics, train_step_loss=train_loss)

    @staticmethod
    def _log_step(train_metrics: TrainMetrics, eval_metrics: EvalMetrics, exported: bool) -> None:
        env_view = ", ".join(
            f"{env}:{score:.3f}" for env, score in sorted(eval_metrics.env_scores.items())
        )
        comp_view = ", ".join(
            f"{env}:{c:.2%}" for env, c in sorted(eval_metrics.env_completeness.items())
        )
        print(
            f"[step={train_metrics.step}] "
            f"loss={train_metrics.loss:.4f} "
            f"avg_reward={train_metrics.avg_reward:.4f} "
            f"avg_kl={train_metrics.avg_kl:.4f} "
            f"gm={eval_metrics.geometric_mean:.4f} "
            f"eligible={eval_metrics.eligible_for_scoring} "
            f"exported={exported} "
            f"envs=[{env_view}] "
            f"completeness=[{comp_view}]"
        )

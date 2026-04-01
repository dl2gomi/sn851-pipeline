from collections import defaultdict
from statistics import mean
from typing import List, Protocol

from ...config import TrainingConfig
from ...schemas import TrainMetrics, Trajectory


class TrainerBackend(Protocol):
    def train_step(self, batch: List[Trajectory], training: TrainingConfig) -> float:
        ...


class Trainer:
    """Training controller that requires a real backend implementation."""

    def __init__(
        self,
        lr: float,
        batch_size: int,
        ppo_epochs: int,
        training_config: TrainingConfig,
        backend: TrainerBackend,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.training_config = training_config
        self.backend = backend

    def apply_optimizer_step(self, batch: List[Trajectory]) -> float:
        return self.backend.train_step(batch=batch, training=self.training_config)

    def update(self, step: int, batch: List[Trajectory]) -> TrainMetrics:
        if not batch:
            return TrainMetrics(
                step=step,
                loss=0.0,
                avg_reward=0.0,
                avg_raw_score=0.0,
                avg_kl=0.0,
                by_env_avg_reward={},
            )

        avg_reward = mean(t.reward for t in batch)
        avg_raw = mean(t.raw_score for t in batch)
        avg_kl = mean(t.kl_estimate for t in batch)
        loss = self.apply_optimizer_step(batch)
        by_env = defaultdict(list)
        for t in batch:
            by_env[t.task.env].append(t.reward)
        return TrainMetrics(
            step=step,
            loss=loss,
            avg_reward=avg_reward,
            avg_raw_score=avg_raw,
            avg_kl=avg_kl,
            by_env_avg_reward={k: mean(v) for k, v in by_env.items()},
        )

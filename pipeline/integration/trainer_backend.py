from typing import List

from ..config import TrainingConfig
from ..core.training.trainer import TrainerBackend
from ..schemas import Trajectory
from .runtime import SharedModelRuntime


class AffineTrainerBackend(TrainerBackend):
    """Training-only adapter; delegates to shared model runtime."""

    def __init__(self, runtime: SharedModelRuntime):
        self.runtime = runtime

    def train_step(self, batch: List[Trajectory], training: TrainingConfig) -> float:
        return self.runtime.train_step(batch=batch, training=training)

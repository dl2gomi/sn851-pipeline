from dataclasses import dataclass
from typing import List, Protocol

from ...schemas import Task, Trajectory


class Policy(Protocol):
    def generate(self, task: Task) -> str:
        ...


@dataclass
class EnvResult:
    score: float
    kl_estimate: float
    is_timeout: bool
    is_format_valid: bool


class EnvironmentExecutor(Protocol):
    def evaluate(self, task: Task, response: str) -> EnvResult:
        ...


class RolloutWorker:
    def __init__(self, policy: Policy, executor: EnvironmentExecutor):
        self.policy = policy
        self.executor = executor

    def rollout(self, tasks: List[Task]) -> List[Trajectory]:
        trajectories: List[Trajectory] = []
        for task in tasks:
            response = self.policy.generate(task)
            env_result = self.executor.evaluate(task, response)
            trajectories.append(
                Trajectory(
                    task=task,
                    response=response,
                    raw_score=env_result.score,
                    kl_estimate=env_result.kl_estimate,
                    is_timeout=env_result.is_timeout,
                    is_format_valid=env_result.is_format_valid,
                )
            )
        return trajectories

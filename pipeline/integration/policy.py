from ..core.execution.rollout import Policy
from ..schemas import Task
from .runtime import SharedModelRuntime


class AffinePolicyAdapter(Policy):
    """Inference-only adapter; delegates to shared model runtime."""

    def __init__(self, runtime: SharedModelRuntime):
        self.runtime = runtime

    def generate(self, task: Task) -> str:
        text, _, _ = self.runtime.rollout_forward(task)
        return text

    def rollout_forward(self, task: Task) -> tuple[str, float, float]:
        """(response text, completion log-prob sum, value estimate at prompt)."""
        return self.runtime.rollout_forward(task)

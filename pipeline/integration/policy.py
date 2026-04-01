from ..core.execution.rollout import Policy
from ..schemas import Task
from .runtime import SharedModelRuntime


class AffinePolicyAdapter(Policy):
    """Inference-only adapter; delegates to shared model runtime."""

    def __init__(self, runtime: SharedModelRuntime):
        self.runtime = runtime

    def generate(self, task: Task) -> str:
        return self.runtime.generate(task)

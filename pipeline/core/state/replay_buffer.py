from collections import defaultdict
from typing import Dict, List, Set

from ...schemas import Trajectory


class ReplayBuffer:
    """FIFO store of trajectories. Oldest entries drop when over ``capacity``.

    For PPO, ``latest(rollout_batch_size)`` should match the most recent rollout
    collection (typically the tail added at the current orchestrator step).
    """

    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self._items: List[Trajectory] = []
        self._completed_task_ids: Dict[str, Set[int]] = defaultdict(set)

    def add_many(self, trajectories: List[Trajectory]) -> None:
        self._items.extend(trajectories)
        for t in trajectories:
            self._completed_task_ids[t.task.env].add(t.task.task_id)
        overflow = len(self._items) - self.capacity
        if overflow > 0:
            self._items = self._items[overflow:]

    def latest(self, n: int) -> List[Trajectory]:
        """Up to ``n`` most recently added trajectories (shallow slice from the end)."""
        if n <= 0:
            return []
        return self._items[-n:]

    def completed_task_ids(self) -> Dict[str, Set[int]]:
        return {k: set(v) for k, v in self._completed_task_ids.items()}

    def __len__(self) -> int:
        return len(self._items)

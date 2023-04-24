import numpy as np
import dataclasses

from typing import Any, Optional, Type
from ReplayTables._utils.logger import logger
from ReplayTables._utils.MinMaxHeap import MinMaxHeap
from ReplayTables.ReplayBuffer import ReplayBufferInterface, T

@dataclasses.dataclass
class PrioritizedHeapConfig:
    threshold: float = 1.0

class PrioritizedHeap(ReplayBufferInterface[T]):
    def __init__(self, max_size: int, structure: Type[T], rng: np.random.RandomState, config: Optional[PrioritizedHeapConfig] = None):
        super().__init__(max_size, structure, rng)

        self._c = config or PrioritizedHeapConfig()
        self._heap = MinMaxHeap[int]()

    def add(self, transition: T, /, priority: float, **kwargs: Any):
        if priority < self._c.threshold:
            return -1

        if self.size() == self._max_size and priority < self._heap.min()[0]:
            return -1

        idx = super().add(transition, **kwargs)
        if self.size() == self._max_size:
            p, _ = self._heap.pop_min()
            logger.debug(f'Heap is full. Tossing priority: {p}')

        self._heap.add(priority, idx)
        return idx

    def _pop_idx(self):
        if self._heap.size() == 0:
            return None

        p, idx = self._heap.pop_max()
        logger.debug(f'Grabbed sample with priority: {p}')
        return idx

    def pop(self):
        idx = self._pop_idx()
        if idx is None:
            return None

        d = self._storage[idx]
        del self._storage[idx]
        return d

    def _sample_idxs(self, n: int):
        idxs = (self._pop_idx() for _ in range(n))
        idxs = (d for d in idxs if d is not None)
        return np.fromiter(idxs, dtype=np.int64)

    def _isr_weights(self, idxs: np.ndarray) -> np.ndarray:
        return np.ones(len(idxs))

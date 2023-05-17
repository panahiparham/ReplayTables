import numpy as np
import dataclasses

from typing import cast, Any, Optional, Tuple, Type
from ReplayTables._utils.logger import logger
from ReplayTables._utils.MinMaxHeap import MinMaxHeap
from ReplayTables.ReplayBuffer import ReplayBufferInterface, EID, EIDS, T

@dataclasses.dataclass
class PrioritizedHeapConfig:
    threshold: float = 1.0

class PrioritizedHeap(ReplayBufferInterface[T]):
    def __init__(self, max_size: int, structure: Type[T], rng: np.random.Generator, config: Optional[PrioritizedHeapConfig] = None):
        super().__init__(max_size, structure, rng)

        self._c = config or PrioritizedHeapConfig()
        self._heap = MinMaxHeap[EID]()

    def size(self):
        return self._heap.size()

    def _add(self, transition: T):
        eid = getattr(transition, 'eid', None)
        if eid is not None:
            eid = cast(EID, eid)
        else:
            eid = cast(EID, self._t)
            self._t += 1
            try:
                setattr(transition, 'eid', eid)
            except Exception: ...

        self._storage[eid] = transition
        return eid

    def add(self, transition: T, /, **kwargs: Any):
        priority = kwargs['priority']
        if priority < self._c.threshold:
            return -1

        if self.size() == self._max_size and priority < self._heap.min()[0]:
            return -1

        eid = self._add(transition)
        if self.size() == self._max_size:
            p, tossed_eid = self._heap.pop_min()
            logger.debug(f'Heap is full. Tossing item: {tossed_eid} - {p}')
            del self._storage[tossed_eid]

        logger.debug(f'Adding element: {eid} - {priority}')
        self._heap.add(priority, eid)
        return eid

    def _pop_min_idx(self):
        if self._heap.size() == 0:
            return None

        p, idx = self._heap.pop_min()
        logger.debug(f'Grabbed sample: {idx} - {p}')
        return idx

    def _pop_idx(self):
        if self._heap.size() == 0:
            return None

        p, idx = self._heap.pop_max()
        logger.debug(f'Grabbed sample: {idx} - {p}')
        return idx

    def pop_min(self):
        idx = self._pop_min_idx()
        if idx is None:
            return None

        d = self._storage[idx]
        del self._storage[idx]
        return d

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

    def sample(self, n: int) -> Tuple[T, EIDS, np.ndarray]:
        batch, idxs, weights = super().sample(n)

        for idx in idxs:
            del self._storage[idx]

        return batch, idxs, weights

    def _isr_weights(self, idxs: EIDS) -> np.ndarray:
        return np.ones(len(idxs))

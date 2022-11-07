from typing import Generic, TypeVar, Type
import numpy as np
from ReplayTables._utils.Distributions import UniformDistribution

T = TypeVar('T')


class ReplayBuffer(Generic[T]):
    def __init__(self, max_size: int, structure: Type[T], rng: np.random.RandomState):
        self._max_size = max_size
        self._structure = structure
        self._rng = rng

        self._t = 0
        self._idx_dist = UniformDistribution(0)
        self._storage = {}

    def size(self):
        return len(self._storage)

    def _update_dist(self, idx: int, transition: T):
        self._idx_dist.update(self.size())

    def add(self, transition: T):
        idx = self._t % self._max_size
        self._t += 1

        self._update_dist(idx, transition)
        self._storage[idx] = transition
        return idx

    def _sample_idxs(self, n: int):
        return self._idx_dist.sample(self._rng, n)

    def _isr_weights(self, idxs: np.ndarray):
        return np.ones(len(idxs))

    def sample(self, n: int):
        idxs = self._sample_idxs(n)

        samples = (self._storage[i] for i in idxs)
        stacked = (np.stack(xs, axis=0) for xs in zip(*samples))
        weights = self._isr_weights(idxs)

        return self._structure(*stacked), idxs, weights

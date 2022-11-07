import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Type
from ReplayTables.ReplayBuffer import ReplayBufferInterface, T
from ReplayTables._utils.Distributions import MixinUniformDistribution, MixtureDistribution, PrioritizedDistribution, SubDistribution, UniformDistribution

@dataclass
class PERConfig:
    new_priority_mode: str = 'max'
    uniform_probability: float = 1e-3
    priority_exponent: float = 0.5
    max_decay: float = 1.

class PrioritizedReplay(ReplayBufferInterface[T]):
    def __init__(self, max_size: int, structure: Type[T], rng: np.random.RandomState, config: Optional[PERConfig] = None):
        super().__init__(max_size, structure, rng)

        self._c = config or PERConfig()
        self._target = UniformDistribution(max_size)

        p = 1 - self._c.uniform_probability
        self._idx_dist = MixtureDistribution(max_size, dists=[
            SubDistribution(d=PrioritizedDistribution, p=p),
            SubDistribution(d=MixinUniformDistribution, p=self._c.uniform_probability),
        ])

        self._max_priority = 1e-16

    def _sample_idxs(self, n: int) -> np.ndarray:
        idxs = self._idx_dist.sample(self._rng, n)
        return np.asarray(idxs)

    def _update_dist(self, idx: int, /, **kwargs: Any):
        if self._c.new_priority_mode == 'max':
            priority = self._max_priority
        elif self._c.new_priority_mode == 'mean':
            priority = self._idx_dist._tree.dim_total(0) / self.size()
            if priority == 0:
                priority = 1e-16
        else:
            raise NotImplementedError()

        idxs = np.array([idx])
        priorities = np.array([priority])
        self._idx_dist.dists[1].update(idxs, priorities)
        self._idx_dist.dists[0].update(idxs, priorities)

    def _isr_weights(self, idxs: np.ndarray):
        return self._idx_dist.isr(self._target, idxs)

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray):
        priorities = priorities ** self._c.priority_exponent
        self._idx_dist.dists[0].update(idxs, priorities)

        self._max_priority = max(
            self._c.max_decay * self._max_priority,
            priorities.max(),
        )

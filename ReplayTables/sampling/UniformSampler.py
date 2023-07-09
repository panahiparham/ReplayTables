import numpy as np
from typing import Any
from ReplayTables.sampling.IndexSampler import IndexSampler
from ReplayTables.interface import IDX, IDXs, Timestep, Batch
from ReplayTables._utils.SamplableSet import SamplableSet

class UniformSampler(IndexSampler):
    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)
        self._non_term = SamplableSet(rng)

    def replace(self, idx: IDX, transition: Timestep, /, **kwargs: Any) -> None:
        if not transition.terminal:
            self._non_term.add(idx)

    def update(self, idxs: IDXs, batch: Batch, /, **kwargs: Any) -> None:
        ...

    def isr_weights(self, idxs: IDXs):
        return np.ones(len(idxs))

    def sample(self, n: int) -> IDXs:
        idxs: Any = self._non_term.sample(n)
        return idxs

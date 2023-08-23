import numpy as np
from typing import Any, Set
from ReplayTables.sampling.IndexSampler import IndexSampler
from ReplayTables.interface import IDX, IDXs, LaggedTimestep, Batch
from ReplayTables._utils.jit import try2jit

class BackwardsSampler(IndexSampler):
    def __init__(
        self,
        jump: int,
        reset_probability: float,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(rng)
        self._reset = reset_probability
        self._jump = jump
        self._batch_size: int | None = None
        self._prior_idxs: IDXs | None = None
        self._size = 0

        self._terminal = set[int]()
        # numba needs help with type inference
        # so add a dummy value to the set
        self._terminal.add(-1)

    def replace(self, idx: IDX, transition: LaggedTimestep, /, **kwargs: Any) -> None:
        self._size = max(self._size, idx + 1)

        self._terminal.discard(idx)
        if transition.terminal:
            self._terminal.add(idx)

    def update(self, idxs: IDXs, batch: Batch, /, **kwargs: Any) -> None:
        ...

    def isr_weights(self, idxs: IDXs):
        return np.ones(len(idxs))

    def sample(self, n: int) -> IDXs:
        if self._prior_idxs is None or self._batch_size != n:
            idxs: Any = self._rng.integers(0, self._size + 1, size=n)
            self._prior_idxs = idxs
            self._batch_size = n
            return idxs

        idxs = _get_predecessors(
            self._rng,
            self._prior_idxs,
            self._size,
            self._reset,
            self._jump,
            self._terminal,
        )
        self._prior_idxs = idxs
        return idxs

@try2jit()
def _get_predecessors(rng: np.random.Generator, prior: IDXs, size: int, reset: float, jump: int, terms: Set[int]):
    n = len(prior)
    idxs: Any = rng.integers(0, size, size=n)

    mask = rng.random(size=n) < reset

    predecessors = prior
    for i in range(n):
        for _ in range(jump):
            predecessors[i] = (predecessors[i] - 1) % size
            if predecessors[i] in terms:
                mask[i] = 1
                break

    new_idxs = mask * idxs + (1 - mask) * predecessors
    return new_idxs.astype(np.uint64)

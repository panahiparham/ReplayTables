import numpy as np

from typing import Any, Set
from numba.typed import List as NList
from ReplayTables.sampling.PrioritySampler import PrioritySampler
from ReplayTables.interface import IDX, Batch, IDXs, LaggedTimestep
from ReplayTables._utils.jit import try2jit

class PrioritySequenceSampler(PrioritySampler):
    def __init__(
        self,
        uniform_probability: float,
        trace_decay: float,
        trace_depth: int,
        combinator: str,
        max_size: int,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(uniform_probability, max_size, rng)

        self._terminal = set[int]()
        # numba needs help with type inference
        # so add a dummy value to the set
        self._terminal.add(-1)

        assert combinator in ['max', 'sum']
        self._combinator = combinator
        self._trace = np.cumprod(np.ones(trace_depth) * trace_decay)
        self._size = 0

    def replace(self, idx: IDX, transition: LaggedTimestep, /, **kwargs: Any) -> None:
        self._size = max(idx + 1, self._size)
        self._terminal.discard(idx)
        if transition.terminal:
            self._terminal.add(idx)

        return super().replace(idx, transition, **kwargs)

    def update(self, idxs: IDXs, batch: Batch, /, **kwargs: Any) -> None:
        priorities = kwargs['priorities']
        self._uniform.update(idxs)

        d = self._p_dist.dim
        tree = self._p_dist.tree.raw

        u_idxs, u_priorities = _update(
            tree,
            d,
            self._size,
            idxs,
            priorities,
            self._combinator,
            self._trace,
            self._terminal,
        )
        self._p_dist.update(u_idxs, u_priorities)


@try2jit()
def _update(tree: NList[np.ndarray], d: int, size: int, idxs: np.ndarray, priorities: np.ndarray, comb: str, trace: np.ndarray, terms: Set[int]):
    depth = len(trace)
    out_idxs = np.empty(depth * len(idxs), dtype=np.uint64)
    out = np.empty(depth * len(idxs))

    def c(a: float, b: float):
        if comb == 'sum':
            return a + b
        return max(a, b)

    j = 0
    for idx, v in zip(idxs, priorities):
        for i in range(depth):
            s_idx = (idx - (i + 1)) % size
            if s_idx in terms: break

            prior = tree[0][d, s_idx]
            new = c(prior, trace[i] * v)

            out_idxs[j] = s_idx
            out[j] = new
            j += 1

    return (
        np.concatenate((idxs, out_idxs[:j])).astype(np.uint64),
        np.concatenate((priorities, out[:j])),
    )

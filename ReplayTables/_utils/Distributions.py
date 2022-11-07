from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Any, Optional, NamedTuple, Sequence, Type
from ReplayTables._utils.SumTree import SumTree

class Distribution:
    def probs(self, idxs: npt.ArrayLike):
        raise NotImplementedError('Expected probs to be implemented')

    def sample(self, rng: np.random.RandomState, n: int):
        raise NotImplementedError('Expected sample to be implemented')

    def isr(self, target: Distribution, idxs: np.ndarray):
        return target.probs(idxs) / self.probs(idxs)


class UniformDistribution(Distribution):
    def __init__(self, size: int):
        super().__init__()

        self._size = size

    def update(self, size: int):
        self._size = size

    def sample(self, rng: np.random.RandomState, n: int):
        if self._size == 1:
            return np.zeros(n)

        return rng.randint(0, self._size, size=n)

    def probs(self, idxs: npt.ArrayLike):
        return np.full_like(idxs, fill_value=(1 / self._size), dtype=np.float_)


class PrioritizedDistribution(Distribution):
    def __init__(self, tree: SumTree, dim: int, config: Optional[Any] = None):
        self._tree = tree
        self._dim = dim

        self._weights = np.zeros(tree.dims)
        self._weights[dim] = 1
        self._config = config

    def probs(self, idxs: npt.ArrayLike):
        idxs = np.asarray(idxs)

        t = self._tree.dim_total(self._dim)
        if t == 0:
            return 0

        v = self._tree.get_values(self._dim, idxs)
        return v / t

    def sample(self, rng: np.random.RandomState, n: int):
        return self._tree.sample(rng, n, self._weights)

    @property
    def dim(self):
        return self._dim

    def update(self, idxs: np.ndarray, values: np.ndarray):
        self._tree.update(self.dim, idxs, values)


class MixinUniformDistribution(PrioritizedDistribution):
    def __init__(self, tree: SumTree, dim: int, config: Optional[Any] = None):
        super().__init__(tree, dim, config)

    def update(self, idxs: np.ndarray, *args, **kwargs):
        self._tree.update(self.dim, idxs, np.ones(len(idxs)))


class SubDistribution(NamedTuple):
    d: Type[PrioritizedDistribution]
    p: float
    config: Optional[Type[Any]] = None
    isr: bool = True


class MixtureDistribution(Distribution):
    def __init__(self, size: int, dists: Sequence[SubDistribution]):
        super().__init__()

        self._dims = len(dists)
        self._tree = SumTree(size, self._dims)

        self.dists = [sub.d(self._tree, i, sub.config) for i, sub in enumerate(dists)]
        self._weights = np.array([sub.p for sub in dists])
        self._mask = np.array([sub.isr for sub in dists], dtype=bool)

        self._fast_isr = np.all(self._mask)

    def _filter_defunct(self):
        # if a distribution ends up with 0 total, it is defunct and cannot be sampled
        # by default the SumTree then puts all probability mass on the final index
        # which isn't great.
        totals = self._tree.all_totals()
        mask = totals != 0

        if np.all(mask):
            return self._weights

        new_weights = self._weights * mask
        w = new_weights / new_weights.sum()
        return w

    def probs(self, idxs: npt.ArrayLike):
        w = self._filter_defunct()
        sub = np.array([d.probs(idxs) for d in self.dists])
        p = w.dot(sub)
        return p

    def sample(self, rng: np.random.RandomState, n: int):
        w = self._filter_defunct()
        w = w * self._tree.effective_weights()
        return self._tree.sample(rng, n, w)

    def isr(self, target: Distribution, idxs: np.ndarray):
        tops = target.probs(idxs)
        subs = np.array([d.probs(idxs) for d in self.dists])
        w = self._filter_defunct()

        # in the base case, we can skip the extra compute
        if self._fast_isr:
            bottoms = w.dot(subs)
            return tops / bottoms

        w = w * self._mask
        missing = 1 - w.sum()
        rest = np.full_like(tops, fill_value=(1 / self._tree.size))

        bottoms = w.dot(subs) + missing * rest
        return tops / bottoms

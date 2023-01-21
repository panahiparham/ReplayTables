from typing import Iterable, Optional
import numpy as np
from numba.typed import List as NList
from ReplayTables._utils.jit import try2jit, try2vectorize
from ReplayTables._utils.MemoryWriter import MemoryWriter, ThreadedWriter

W = Optional[np.ndarray]


@try2jit
def _nearestPowerOf2(x: float):
    i = np.log2(x)
    u = np.ceil(i)
    return int(2**u)

@try2jit
def _update(tree: NList[np.ndarray], dim: int, idxs: np.ndarray, values: np.ndarray):
    for idx, value in zip(idxs, values):
        sub_idx = idx
        old = tree[0][dim, idx]

        for i in range(len(tree)):
            tree[i][dim, sub_idx] += value - old
            sub_idx = int(sub_idx // 2)

@try2vectorize
def _bound(x: int, ma: int):
    return min(x, ma)

@try2jit
def _query(tree: NList[np.ndarray], size: int, weights: np.ndarray, values: np.ndarray):
    totals = np.zeros(len(values))
    idxs = np.zeros(len(values), dtype=np.int64)
    for i in range(len(tree) - 2, -1, -1):
        layer = tree[i]

        idxs = idxs * 2
        lefts = weights.dot(layer[:, idxs])
        rights = weights.dot(layer[:, idxs + 1])
        mask = lefts <= (values - totals)
        mask &= rights > 0
        totals = totals + mask * lefts
        idxs = idxs + mask

    return _bound(idxs, size - 1)


class SumTree:
    def __init__(self, size: int, dims: int = 1, fast_mode: bool = False, _defer_build: bool = False):
        self._size = size
        self._dims = dims

        self._total_size = _nearestPowerOf2(size)

        layers = []
        if not _defer_build:
            for i in range(int(np.log2(self._total_size)), -1, -1):
                layers.append(np.zeros((dims, 2**i)))

        self._tree: NList[np.ndarray] = NList(layers)

        self._writer = ThreadedWriter(self._tree, dims) if fast_mode else MemoryWriter(self._tree, dims)
        # cached to avoid recreating this space in memory repeatedly
        self._u = np.ones(dims)

    @property
    def dims(self):
        return self._dims

    @property
    def size(self):
        return self._size

    def update(self, dim: int, idxs: Iterable[int], values: Iterable[float]):
        idxs = np.asarray(idxs)
        values = np.asarray(values)

        self._writer.write(dim, idxs, values)

    def get_value(self, dim: int, idx: int) -> float:
        return self._tree[0][dim, idx]

    def get_values(self, dim: int, idxs: np.ndarray) -> np.ndarray:
        return self._tree[0][dim, idxs]

    def dim_total(self, dim: int) -> float:
        return self._tree[-1][dim, 0]

    def all_totals(self) -> np.ndarray:
        return self._tree[-1][:, 0]

    def total(self, w: W = None) -> float:
        w_ = self._get_w(w)
        return w_.dot(self._tree[-1])[0]

    def effective_weights(self):
        t = self.all_totals()
        return _safe_invert(t)

    def sample(self, rng: np.random.RandomState, n: int, w: W = None) -> np.ndarray:
        w_ = self._get_w(w)
        t = self.total(w_)
        assert t > 0, "Cannot sample when the tree is empty or contains negative values"

        rs = rng.uniform(0, t, size=n)
        return _query(self._tree, self._size, w_, rs)

    def sync(self):
        self._writer.sync()

    def _get_w(self, w: W = None):
        if w is None:
            return self._u

        return w

    def __getstate__(self):
        return {
            'size': self._size,
            'dims': self._dims,
            'memory': list(self._tree)
        }

    def __setstate__(self, state):
        self.__init__(state['size'], state['dims'], _defer_build=True)
        self._tree = NList(state['memory'])
        self._writer._mem = self._tree

@try2jit
def _safe_invert(arr: np.ndarray):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = 0 if arr[i] == 0 else 1 / arr[i]

    return out

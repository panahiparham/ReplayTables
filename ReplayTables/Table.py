import numpy as np
import numpy.typing as npt
from typing import Dict, Iterable, List, Sequence, Tuple, TypedDict
from numba import njit
from ReplayTables.RandDict import RandDict

class _ColumnDefReq(TypedDict):
    name: str
    shape: npt._ShapeLike
    dtype: npt.DTypeLike

class ColumnDef(_ColumnDefReq, total=False):
    pad: float

def asTuple(shape: npt._ShapeLike) -> Tuple[int, ...]:
    if isinstance(shape, tuple):
        return shape

    if isinstance(shape, list):
        return tuple(shape)

    if isinstance(shape, int):
        return (shape, )

    raise Exception("Could not cast shape to tuple!")

class Table:
    def __init__(self, max_size: int, columns: Sequence[ColumnDef], seed: int = 0):
        self.max_size = max_size
        self._column_defs = columns

        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # monotonically increasing index which counts how many times we've added
        # technically we rely on numpy.int64 in the code so there is a limit
        # but that's okay, this limit is too huge for my use cases
        self._idx = 0
        self._samples = 0

        # force a consistent order over columns
        # defined by user input order
        self._col_names = list(map(lambda c: c['name'], columns))
        self.columns: Dict[str, np.ndarray] = {}

        # views of this table
        # this need to be informed whenever data is added
        # or whenever a trajectory terminates
        self._subscribers: List[View] = []

        # values to pad a tensor with
        # depends on datatype
        self.pads: List[float] = []

        # build these at the end in an easily overrideable function
        self._buildColumns()

    def _buildColumns(self):
        for col_def in self._column_defs:
            # construct the shape of the storage
            # which should be the shape of the column, plus
            # a leading axis of size max_size
            shape = (self.max_size, ) + asTuple(col_def['shape'])

            # it's okay to use totally empty arrays and not waste time
            # cleaning memory. We will do bound checks and avoid
            # reaching into uninitialized memory
            column = np.empty(shape, dtype=col_def['dtype'])
            self.columns[col_def['name']] = column

            # figure out what value to use to pad arrays
            if 'pad' in col_def:
                self.pads.append(col_def['pad'])
            elif np.issubdtype(col_def['dtype'], np.integer):
                self.pads.append(0)
            else:
                self.pads.append(np.nan)

    def addSubscriber(self, sus: "View"):
        if self._idx > 0:
            raise Exception("Cannot subscribe after data has already been collected")

        self._subscribers.append(sus)

    def addTuple(self, data: Sequence[npt.ArrayLike]):
        for i, name in enumerate(self._col_names):
            col = self.columns[name]
            d = data[i]

            col[self._idx % self.max_size] = d

        for sus in self._subscribers: sus._onAdd(self._idx)
        self._idx += 1
        self._samples = min(self._samples + 1, self.max_size)

    def endTrajectory(self):
        for sus in self._subscribers: sus._onEnd()

    def _iterCols(self):
        return (self.columns[name] for name in self._col_names)

    def getSequence(self, seq: np.ndarray, pad: int = 0) -> Tuple[np.ndarray, ...]:
        if not pad:
            return tuple((col[seq] for col in self._iterCols()))
        else:
            return tuple((padded(col[seq], pad, pad_val) for col, pad_val in zip(self._iterCols(), self.pads)))

    def getAll(self):
        return tuple((np.roll(col, -self._idx, axis=0) for col in self._iterCols()))

    def sample(self, size: int = 1):
        idxs = self._rng.permutation(self._samples)[:size]
        return tuple((col[idxs] for col in self._iterCols()))

class View:
    def __init__(self, table: Table, size: int):
        self.max_age = table.max_size
        self.size = size

        self._table = table
        self._table.addSubscriber(self)

        self._refs: RandDict[int, Tuple[int, int]] = RandDict()
        self._idx = 0

        # track the current sequence of indices
        self._seq_idx = 0

    def _onAdd(self, idx: int):
        self._refs[self._idx] = (idx, idx)
        self._idx += 1
        self._seq_idx += 1

        n = min(self.size, self._seq_idx)
        to_update = (i - 1 for i in range(self._idx, self._idx - n, -1))

        for i in to_update:
            self._refs[i] = (self._refs[i][0], idx + 1)

    def _onEnd(self):
        self._seq_idx = 0

    def _seq2TensorTuple(self, seqs: Iterable[Tuple[int, int]]):
        cols = (self._table.getSequence(rotatedSequence(seq[0], seq[1], self._table.max_size), self.size) for seq in seqs)

        return tuple(map(np.stack, zip(*cols)))

    def _resample(self) -> Tuple[int, int]:
        idx = self._table._rng.randint(0, len(self._refs))
        seq = self._refs.getIndex(idx)

        age = self._table._idx - seq[0]
        if age > self.max_age:
            self._refs.delIndex(idx)
            return self._resample()

        return seq

    def sample(self, size: int = 1):
        seqs = (self._resample() for _ in range(size))
        return self._seq2TensorTuple(seqs)

    def getAll(self):
        self.clearOld()
        return self._seq2TensorTuple(self._refs.values())

    def clearOld(self):
        def to_del():
            for key in self._refs:
                seq = self._refs[key]

                if self._table._idx - seq[0] > self.max_age:
                    yield key

        # note this needs to be 2 loops
        # otherwise we change dict while iterating, which is error-prone
        keys = list(to_del())
        for key in keys:
            del self._refs[key]

@njit(cache=True)
def rotatedSequence(lo: int, hi: int, mod: int) -> np.ndarray:
    seq = np.arange(lo, hi, dtype=np.int64)
    return seq % mod

@njit(cache=True)
def padded(arr: np.ndarray, size: int, value: float = np.nan):
    out = np.ones((size, ) + arr.shape[1:], dtype=arr.dtype) * value
    out[:arr.shape[0]] = arr
    return out

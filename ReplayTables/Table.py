import numpy as np
import numpy.typing as npt
from typing import Dict, Iterable, List, Sequence, Tuple, TypedDict, Union, SupportsIndex, cast
from ReplayTables._utils.RandDict import RandDict
from ReplayTables._utils.jit import try2jit

ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]

class _ColumnDefReq(TypedDict):
    name: str
    shape: ShapeLike

class ColumnDef(_ColumnDefReq, total=False):
    pad: float
    pad_multiple: int
    dtype: npt.DTypeLike

def asTuple(shape: ShapeLike) -> Tuple[int, ...]:
    if isinstance(shape, tuple):
        return shape

    if isinstance(shape, list):
        return cast(Tuple[int, ...], tuple(shape))

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

        # the padding size will always be a multiple of this value
        # useful when the padded data has unknown size and is being shipped
        # off to jax (or other XLA compilers) and we don't want kernel recompilations
        self._multiples: List[int] = []

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
            column = np.empty(shape, dtype=col_def.get('dtype'))
            self.columns[col_def['name']] = column

            if 'pad_multiple' in col_def:
                self._multiples.append(col_def['pad_multiple'])
            else:
                self._multiples.append(1)

            # figure out what value to use to pad arrays
            if 'pad' in col_def:
                self.pads.append(col_def['pad'])
            elif np.issubdtype(col_def.get('dtype'), np.integer):
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
            return tuple((padded(col[seq], pad, mult, pad_val) for col, mult, pad_val in zip(self._iterCols(), self._multiples, self.pads)))

    def getAll(self):
        return tuple((np.roll(col, -self._idx, axis=0) for col in self._iterCols()))

    def sample(self, size: int = 1):
        idxs = self._rng.permutation(self._samples)[:size]
        return tuple((col[idxs] for col in self._iterCols()))

    def __len__(self):
        return self._samples

class View:
    def __init__(self, table: Table, size: int):
        # sequences whose starting index is older than this are invalid
        # we will clear these lazily when able
        self.max_age = table.max_size
        self.size = size

        # add a communication link between this view and its parent table
        # so that states are synchronized
        self._table = table
        self._table.addSubscriber(self)

        # a special dictionary which stores keys and values in hash tables
        # to allow O(1) random sampling (instead of O(n))
        self._refs: RandDict[int, Tuple[int, int]] = RandDict()

        # monotonically increasing index for accessing
        # self._refs
        self._idx = 0

        # track the current sequence of indices
        # reset whenever the trajectory ends
        self._seq_idx = 0

        # keep track of the last sequence length from this view
        # that way we can back-track to find the last element
        self._last_seq_length = -1

    def _onAdd(self, idx: int):
        self._refs[self._idx] = (idx, idx)
        self._idx += 1
        self._seq_idx += 1

        n = min(self.size, self._seq_idx)
        to_update = (i - 1 for i in range(self._idx, self._idx - n, -1))

        for i in to_update:
            self._refs[i] = (self._refs[i][0], idx + 1)

    def _onEnd(self):
        self._last_seq_length = min(self._seq_idx, self.size)
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

    def numSequencesThisRound(self):
        # this happens if we never see a termination
        if self._last_seq_length == -1:
            return min(self._idx, self.size)

        return min(self._last_seq_length, self.size)

    def getLastComplete(self, offset: int = 0):
        # clear out memory only when it is twice as full as necessary
        # save some compute since we are hardly using any memory
        if len(self._refs) > self._table.max_size * 2:
            self.clearOld()

        last = self.numSequencesThisRound()
        seq = self._refs[self._idx - last + offset]
        return self._seq2TensorTuple([seq])

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

@try2jit
def rotatedSequence(lo: int, hi: int, mod: int) -> np.ndarray:
    seq = np.arange(lo, hi, dtype=np.int64)
    return seq % mod

@try2jit
def padded(arr: np.ndarray, size: int, mult: int, value: float = np.nan):
    s = int(np.ceil(size / mult) * mult)
    out = np.ones((s, ) + arr.shape[1:], dtype=arr.dtype) * value
    out[:arr.shape[0]] = arr
    return out

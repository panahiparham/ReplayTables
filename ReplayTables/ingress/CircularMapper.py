import numpy as np
from typing import Any
from ReplayTables.ingress.IndexMapper import IndexMapper
from ReplayTables.interface import EID, IDX, EIDs, IDXs
from ReplayTables._utils.jit import try2jit

class CircularMapper(IndexMapper):
    def __init__(self, max_size: int):
        self._max_size = max_size

    def eid2idx(self, eid: EID) -> IDX:
        idx: Any = eid % self._max_size
        return idx

    def eids2idxs(self, eids: EIDs) -> IDXs:
        idxs: Any = eids % self._max_size
        return idxs

    def add_eid(self, eid: EID, /, **kwargs: Any) -> IDX:
        return self.eid2idx(eid)

    def remove_eid(self, eid: EID) -> IDX:
        return self.eid2idx(eid)

    def eids2idxs_sequence(self, start: EIDs, lag: int) -> IDXs:
        idxs: Any = _eid_to_idx_sequence(start, lag + 1, self._max_size)
        return idxs

@try2jit()
def _eid_to_idx_sequence(start: EIDs, lag: int, max_size: int):
    out = np.empty((lag, len(start)), dtype=np.int32)

    for i in range(lag):
        out[i] = (start + i) % max_size

    return out

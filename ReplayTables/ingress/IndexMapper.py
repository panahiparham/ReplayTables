import numpy as np
from typing import Any
from abc import abstractmethod
from ReplayTables.interface import EID, IDX, EIDs, IDXs

class IndexMapper:
    def __init__(self):
        ...

    @abstractmethod
    def add_eid(self, eid: EID, /, **kwargs: Any) -> IDX: ...

    @abstractmethod
    def remove_eid(self, eid: EID) -> IDX: ...

    @abstractmethod
    def eid2idx(self, eid: EID) -> IDX: ...

    @abstractmethod
    def eids2idxs(self, eids: EIDs) -> IDXs: ...

    def eids2idxs_sequence(self, start: EIDs, lag: int) -> IDXs:
        out: Any = np.empty((lag + 1, len(start)), dtype=np.int32)
        for i in range(lag + 1):
            n_eids: Any = start + i
            out[i] = self.eids2idxs(n_eids)

        return out

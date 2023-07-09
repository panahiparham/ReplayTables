import numpy as np
from typing import Any, Dict
from ReplayTables.interface import Batch, EIDs, Timestep, TaggedTimestep, EID, IDX, IDXs
from ReplayTables.storage.Storage import Storage

from ReplayTables._utils.jit import try2jit

class BasicStorage(Storage):
    def __init__(self, max_size: int, sample_obs: np.ndarray | None = None, action_dtype: Any = np.int_):
        super().__init__(max_size)

        self._state_store: Dict[IDX, np.ndarray] = {}
        self._eids = np.zeros(max_size, dtype=np.uint64)
        self._a = np.zeros(max_size, dtype=action_dtype)
        self._r = np.zeros(max_size)
        self._term = np.zeros(max_size, dtype=np.bool_)
        self._gamma = np.zeros(max_size)

        self._zero_obs: np.ndarray | None = None
        if sample_obs is not None:
            self._zero_obs = np.zeros_like(sample_obs)

    def add(self, idx: IDX, eid: EID, transition: Timestep, /, **kwargs: Any):
        if transition.x is not None:
            self._state_store[idx] = transition.x
        else:
            assert self._zero_obs is not None, 'Need to specify a default sample observation is terminal states are marked as None'
            self._state_store[idx] = self._zero_obs

        self._r[idx] = transition.r
        self._a[idx] = transition.a
        self._term[idx] = transition.terminal
        self._gamma[idx] = transition.gamma
        self._eids[idx] = eid

    def set(self, idx: IDX, eid: EID, transition: Timestep):
        if transition.x is not None:
            self._state_store[idx] = transition.x
        else:
            assert self._zero_obs is not None, 'Need to specify a default sample observation is terminal states are marked as None'
            self._state_store[idx] = self._zero_obs

        self._r[idx] = transition.r
        self._a[idx] = transition.a
        self._term[idx] = transition.terminal
        self._gamma[idx] = transition.gamma
        self._eids[idx] = eid

    def get(self, idxs: IDXs, n_idxs: IDXs, lag: int) -> Batch:
        x = np.stack([self._state_store[idx] for idx in idxs], axis=0)
        xp = np.stack([self._state_store[idx] for idx in n_idxs], axis=0)

        eids: Any = self._eids[idxs]
        r, gamma, term = _return(self._max_size - lag, idxs, lag, self._r, self._term, self._gamma)
        return Batch(
            x=x,
            a=self._a[idxs],
            r=r,
            gamma=gamma,
            terminal=term,
            eid=eids,
            xp=xp,
        )

    def get_item(self, idx: IDX) -> TaggedTimestep:
        eid: Any = self._eids[idx]
        return TaggedTimestep(
            x=self._state_store[idx],
            a=self._a[idx],
            r=self._r[idx],
            gamma=self._gamma[idx],
            terminal=self._term[idx],
            eid=eid,
        )

    def get_eids(self, idxs: IDXs) -> EIDs:
        eids: Any = self._eids[idxs]
        return eids

    def __delitem__(self, idx: IDX):
        del self._state_store[idx]

    def __len__(self):
        return len(self._state_store)

@try2jit()
def _return(max_size: int, idxs: np.ndarray, lag: int, r: np.ndarray, term: np.ndarray, gamma: np.ndarray):
    samples = len(idxs)
    g = np.zeros(samples)
    d = np.ones(samples)
    t = np.zeros(samples, dtype=np.bool_)

    for b in range(samples):
        idx = idxs[b]
        for i in range(lag):
            n_idx = int((idx + i) % max_size)
            g[b] += d[b] * r[n_idx]

            if term[n_idx]:
                t[b] = True
                break

            d[b] *= gamma[n_idx]

    return g, d, t

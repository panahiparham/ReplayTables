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

    def get(self, idx_seqs: IDXs) -> Batch:
        idxs = idx_seqs[0]

        r, gamma, term, n_idxs = _return(idx_seqs, self._r, self._term, self._gamma)

        x = np.stack([self._state_store[idx] for idx in idxs], axis=0)
        xp = np.stack([self._state_store[idx] for idx in n_idxs], axis=0)

        eids: Any = self._eids[idxs]
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
def _return(idx_seqs: np.ndarray, r: np.ndarray, term: np.ndarray, gamma: np.ndarray):
    lag = idx_seqs.shape[0]
    samples = idx_seqs.shape[1]

    g = np.zeros(samples)
    d = np.ones(samples)
    t = np.zeros(samples, dtype=np.bool_)
    n_idxs = idx_seqs[-1]

    for b in range(samples):
        for i in range(1, lag):
            idx = idx_seqs[i, b]
            g[b] += d[b] * r[idx]

            if term[idx]:
                n_idxs[b] = idx
                t[b] = True
                break

            d[b] *= gamma[idx]

    return g, d, t, n_idxs

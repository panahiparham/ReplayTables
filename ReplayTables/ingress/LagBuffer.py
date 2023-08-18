import numpy as np
from ReplayTables._utils.jit import try2jit
from typing import Any, Dict, List, Tuple
from ReplayTables.interface import Timestep, LaggedTimestep, EID

class LagBuffer:
    def __init__(self, lag: int):
        self._lag = lag
        self._max_len = lag + 1
        self._eid: Any = 0
        self._buffer: Dict[int, Tuple[EID, Timestep]] = {}
        self._r = np.zeros(self._max_len, dtype=np.float_)
        self._g = np.zeros(self._max_len, dtype=np.float_)

    def add(self, experience: Timestep):
        idx = self._eid % self._max_len
        self._r[idx] = experience.r
        self._g[idx] = experience.gamma

        eid: Any = self._eid
        self._eid += 1
        f_idx = (eid - self._lag) % self._max_len
        if experience.terminal:
            self._eid -= 1
            eid = None

        self._buffer[idx] = (eid, experience)
        out: List[LaggedTimestep] = []
        if len(self._buffer) <= self._lag:
            return out

        f_eid, f = self._buffer[f_idx]
        r, g = _accumulate_return(self._r, self._g, f_idx, self._lag, self._max_len)

        assert f.x is not None
        out.append(LaggedTimestep(
            eid=f_eid,
            x=f.x,
            a=f.a,
            r=r,
            gamma=g,
            extra=f.extra or {},
            terminal=experience.terminal,
            n_eid=eid,
            n_x=experience.x,
        ))

        if not experience.terminal:
            return out

        for i in range(1, self._lag):
            start = (f_idx + i) % self._max_len
            f_eid, f = self._buffer[start]
            r, g = _accumulate_return(self._r, self._g, start, self._lag - i, self._max_len)

            assert f.x is not None
            out.append(LaggedTimestep(
                eid=f_eid,
                x=f.x,
                a=f.a,
                r=r,
                gamma=g,
                extra=f.extra or {},
                terminal=experience.terminal,
                n_eid=eid,
                n_x=experience.x,
            ))

        self.flush()
        return out

    def flush(self):
        self._buffer = {}
        self._r = np.zeros(self._max_len, dtype=np.float_)
        self._g = np.zeros(self._max_len, dtype=np.float_)


@try2jit()
def _accumulate_return(rs: np.ndarray, gs: np.ndarray, start: int, steps: int, max_len: int):
    g = 1.
    r = 0.
    for i in range(steps):
        idx = (start + i + 1) % max_len
        assert not np.isnan(rs[idx])
        r += rs[idx] * g
        g *= gs[idx]

    return r, g

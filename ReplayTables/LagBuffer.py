from __future__ import annotations
from typing import Any, Deque, Generic, List, NamedTuple, Optional, Protocol, TypeVar
from collections import deque


class Addable(Protocol):
    def __add__(self, other: Any, /) -> Any:
        ...

class Ring(Addable):
    def __add__(self, other: Any, /) -> Any:
        ...

    def __mul__(self, other: Any, /) -> Any:
        ...

    def __rmul__(self, other: Any, /) -> Any:
        ...

    def __pow__(self, other: Any, /) -> Any:
        ...


class Experience(Protocol):
    s: Any
    a: Any
    r: Optional[Ring]
    gamma: Ring
    terminal: bool

class LaggedExperience(NamedTuple):
    s: Any
    a: Any
    r: Any
    gamma: Any
    terminal: bool
    sp: Any


T = TypeVar('T', bound=Experience)
class LagBuffer(Generic[T]):
    def __init__(self, lag: int):
        self._lag = lag
        self._buffer: Deque[T] = deque(maxlen=lag + 1)

    def add(self, experience: T):
        self._buffer.append(experience)

        out: List[LaggedExperience] = []
        if len(self._buffer) <= self._lag:
            return out

        f = self._buffer[0]
        r, g = _accumulate_return(self._buffer, 0)
        out.append(LaggedExperience(
            s=f.s,
            a=f.a,
            r=r,
            gamma=g,
            terminal=experience.terminal,
            sp=experience.s,
        ))

        if not experience.terminal:
            return out

        for i in range(1, self._lag):
            f = self._buffer[i]
            r, g = _accumulate_return(self._buffer, i)
            out.append(LaggedExperience(
                s=f.s,
                a=f.a,
                r=r,
                gamma=g,
                terminal=experience.terminal,
                sp=experience.s,
            ))

        return out

    def flush(self):
        self._buffer.clear()


def _accumulate_return(b: Deque[T], start: int):
    g = 1.
    r = 0.
    for i in range(start + 1, len(b)):
        e = b[i]
        assert e.r is not None
        r += e.r * g
        g *= e.gamma

    return r, g

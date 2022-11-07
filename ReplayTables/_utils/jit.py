import logging
from typing import Any, Callable, TypeVar, Protocol, cast
import numpy.typing as npt

_has_warned = False
T = TypeVar('T', bound=Callable[..., Any])

def try2jit(f: T) -> T:
    try:
        from numba import njit
        return njit(f, cache=True, nogil=True, fastmath=True)
    except Exception:
        global _has_warned
        if not _has_warned:
            _has_warned = True
            logging.getLogger('ReplayTables').warn('Could not jit compile --- expect slow performance')

        return f


class Vectorized(Protocol):
    def __call__(self, *args: npt.ArrayLike) -> npt.ArrayLike:
        ...

def try2vectorize(f: Any) -> Vectorized:
    try:
        from numba import vectorize
        return vectorize(f, cache=True)

    except Exception as e:
        logging.getLogger('ReplayTables').error(e)
        global _has_warned
        if not _has_warned:
            _has_warned = True
            logging.getLogger('ReplayTables').warn('Could not jit compile --- expect slow performance')

        def _inner(arr, *args, **kwargs):
            out = []
            for v in arr:
                out.append(f(v, *args, **kwargs))

            return out

        return cast(Any, _inner)

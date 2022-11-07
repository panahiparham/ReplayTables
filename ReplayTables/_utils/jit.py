import logging
from typing import Any, Callable, TypeVar

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

def try2vectorize(f: T) -> T:
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

        return _inner

import numpy as np

from typing import Dict, Generic, TypeVar, Tuple
from ReplayTables._utils.jit import try2jit

_HEAP = Tuple[np.ndarray, np.ndarray]

T = TypeVar('T')
class MinMaxHeap(Generic[T]):
    def __init__(self) -> None:
        super().__init__()

        self._t = 0
        self._i = np.array(0, dtype=np.uint32)
        self._storage: Dict[int, T] = {}

        self._heap: _HEAP = (
            # actual heap values
            np.zeros(4),
            # storage pointers
            np.zeros(4, dtype=np.uint32),
        )

    def add(self, priority: float, item: T):
        self._extend()

        self._heap[0][self._t] = priority
        self._heap[1][self._t] = self._i
        self._storage[self._i.item()] = item

        self._heap = _push_up(self._heap, self._t)

        self._t += 1
        self._i += np.array(1, dtype=np.uint32)

    def size(self):
        return self._t

    def _extend(self):
        if self._t == self._heap[0].size:
            self._heap = _extend(self._heap)

    def _get(self, i: int):
        p = self._heap[0][i]
        idx = self._heap[1][i]

        return p, self._storage[idx]

    def min(self):
        return self._get(0)

    def _max_i(self):
        p, _ = self._heap

        if self._t == 1:
            return 0

        if self._t == 2:
            return 1

        if p[1] > p[2]:
            return 1

        return 2

    def max(self):
        i = self._max_i()
        return self._get(i)

    def pop_min(self):
        p = self._heap[0][0]
        idx = self._heap[1][0]
        v = self._storage[idx]
        self._replace_with_last(0)

        del self._storage[idx]
        return p, v

    def pop_max(self):
        i = self._max_i()

        p = self._heap[0][i]
        idx = self._heap[1][i]

        v = self._storage[idx]
        self._replace_with_last(i)

        del self._storage[idx]
        return p, v

    def _replace_with_last(self, i: int):
        self._t -= 1
        self._heap = _delete(self._heap, i, self._t)


@try2jit()
def _extend(heap: _HEAP) -> _HEAP:
    data, idxs = heap
    ext_data = np.zeros_like(data)
    ext_idxs = np.zeros_like(idxs)

    return (
        np.concatenate((data, ext_data)),
        np.concatenate((idxs, ext_idxs)),
    )

@try2jit()
def swap(h: _HEAP, i: int, j: int):
    v = h[0][i]
    idx = h[1][i]
    h[0][i] = h[0][j]
    h[1][i] = h[1][j]
    h[0][j] = v
    h[1][j] = idx
    return h

@try2jit(inline='always')
def _is_min_level(i: int):
    level = np.floor(np.log2(i + 1))
    return level % 2 == 0

@try2jit(inline='always')
def parent(i: int):
    return int((i - 1) // 2)

@try2jit(inline='always')
def grandparent(i: int):
    return int((i - 3) // 4)

@try2jit(inline='always')
def child(i: int):
    return 2 * i + 1

@try2jit(inline='always')
def grandchild(i: int):
    return 4 * i + 3

@try2jit(inline='always')
def _has_children(i: int, size: int):
    c = child(i)
    return c < size

@try2jit()
def _delete(h: _HEAP, i: int, size: int):
    # do the replacement
    h[0][i] = h[0][size]
    h[1][i] = h[1][size]

    # wipe away residual
    h[0][size] = 0
    h[1][size] = 0

    # ensure heap property is respected
    return _push_down(h, i, size)

# -------------
# -- Push up --
# -------------

@try2jit()
def _push_up(h: _HEAP, i: int):
    if i == 0:
        return h

    if _is_min_level(i):
        p = parent(i)
        if h[0][i] > h[0][p]:
            h = swap(h, i, p)
            h = _push_up_max(h, p)
        else:
            h = _push_up_min(h, i)

    else:
        p = parent(i)
        if h[0][i] < h[0][p]:
            h = swap(h, i, p)
            h = _push_up_min(h, p)
        else:
            h = _push_up_max(h, i)

    return h

@try2jit()
def _push_up_min(h: _HEAP, i: int):
    while i > 2:
        gp = grandparent(i)
        if h[0][i] < h[0][gp]:
            swap(h, i, gp)
            i = gp
        else:
            return h

    return h

@try2jit()
def _push_up_max(h: _HEAP, i: int):
    while i > 2:
        gp = grandparent(i)
        if h[0][i] > h[0][gp]:
            swap(h, i, gp)
            i = gp
        else:
            return h

    return h


# ---------------
# -- Push down --
# ---------------

@try2jit()
def _push_down(h: _HEAP, m: int, size: int):
    while _has_children(m, size):
        i = m
        if _is_min_level(i):
            m, t = _smallest_child_or_grandchild(h, i, size)
            if h[0][m] < h[0][i]:
                h = swap(h, m, i)
                if t == 'g':
                    p = parent(m)
                    if h[0][m] > h[0][p]:
                        h = swap(h, m, p)
                else:
                    break
            else:
                break

        else:
            m, t = _largest_child_or_grandchild(h, i, size)
            if h[0][m] > h[0][i]:
                h = swap(h, m, i)
                if t == 'g':
                    p = parent(m)
                    if h[0][m] < h[0][p]:
                        h = swap(h, m, p)
                else:
                    break
            else:
                break
    return h

@try2jit()
def _smallest_child_or_grandchild(h: _HEAP, i: int, size: int):
    c = child(i)
    if c + 1 < size and h[0][c + 1] < h[0][c]:
        c = c + 1

    g = grandchild(i)
    if g >= size:
        return c, 'c'

    e = min(size, g + 5)
    g = int(h[0][g:e].argmin()) + g

    if h[0][c] < h[0][g]:
        return c, 'c'

    return g, 'g'

@try2jit()
def _largest_child_or_grandchild(h: _HEAP, i: int, size: int):
    c = child(i)
    if c + 1 < size and h[0][c + 1] > h[0][c]:
        c = c + 1

    g = grandchild(i)
    if g >= size:
        return c, 'c'

    e = min(size, g + 5)
    g = int(h[0][g:e].argmax()) + g
    if h[0][g] > h[0][c]:
        return g, 'g'

    return c, 'c'

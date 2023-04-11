import numpy as np
from abc import abstractmethod
from typing import Any, Dict, Generic, List, NamedTuple, Tuple, TypeVar, Type, Union, cast
from ReplayTables.Distributions import UniformDistribution

T = TypeVar('T', bound=NamedTuple)


class ReplayBufferInterface(Generic[T]):
    def __init__(self, max_size: int, structure: Type[T], rng: np.random.RandomState):
        self._max_size = max_size
        self._structure = cast(Any, structure)
        self._rng = rng

        self._t = 0
        self._storage: Dict[int, T] = {}

        self._views: List[ReplayViewInterface[T]] = []

    def size(self) -> int:
        return len(self._storage)

    def add(self, transition: T, /, **kwargs: Any):
        idx = self._t % self._max_size
        self._t += 1

        self._storage[idx] = transition
        self._update_dist(idx, transition=transition, **kwargs)
        for view in self._views:
            view._update_dist(idx, transition=transition, **kwargs)

        return idx

    def sample(self, n: int) -> Tuple[T, np.ndarray, np.ndarray]:
        idxs = self._sample_idxs(n)
        weights = self._isr_weights(idxs)
        return self.get(idxs), idxs, weights

    def get(self, idxs: np.ndarray):
        samples = (self._storage[i] for i in idxs)
        stacked = (np.stack(xs, axis=0) for xs in zip(*samples))

        return self._structure(*stacked)

    def register_view(self, view: Any):
        self._views.append(view)

    def backwards_idxs(self, idxs: np.ndarray):
        t = self._t - 1
        return (t - idxs) % self.size()

    def forwards_idxs(self, idxs: np.ndarray):
        t = self._t - 1
        return ((t - self.size()) + idxs) % self.size()

    def __getitem__(self, idx: Union[int, slice]):
        # 0 means the newest element
        # -1 means the oldest element

        if isinstance(idx, int):
            idxs = np.array([idx])
        else:
            idxs = np.arange(idx.start, idx.stop, idx.step, dtype=np.int32)

        idxs = self.backwards_idxs(idxs)
        weights = self._isr_weights(idxs)
        return self.get(idxs), idxs, weights

    # required private methods
    @abstractmethod
    def _sample_idxs(self, n: int) -> np.ndarray: ...

    @abstractmethod
    def _isr_weights(self, idxs: np.ndarray) -> np.ndarray: ...

    # optional methods
    def _update_dist(self, idx: int, /, **kwargs: Any): ...


class ReplayBuffer(ReplayBufferInterface[T]):
    def __init__(self, max_size: int, structure: Type[T], rng: np.random.RandomState):
        super().__init__(max_size, structure, rng)
        self._idx_dist = UniformDistribution(0)

    def _update_dist(self, idx: int, /, **kwargs: Any):
        self._idx_dist.update(self.size())

    def _sample_idxs(self, n: int):
        return self._idx_dist.sample(self._rng, n)

    def _isr_weights(self, idxs: np.ndarray):
        return np.ones(len(idxs))

class ReplayViewInterface(Generic[T]):
    def __init__(self, buffer: ReplayBufferInterface[T]):
        self._buffer = buffer
        self._rng = buffer._rng

        self._buffer.register_view(self)

    def size(self) -> int:
        return self._buffer.size()

    def sample(self, n: int) -> Tuple[T, np.ndarray, np.ndarray]:
        idxs = self._sample_idxs(n)
        weights = self._isr_weights(idxs)
        return self._buffer.get(idxs), idxs, weights

    # required private methods
    @abstractmethod
    def _sample_idxs(self, n: int) -> np.ndarray: ...

    @abstractmethod
    def _isr_weights(self, idxs: np.ndarray) -> np.ndarray: ...

    # optional methods
    def _update_dist(self, idx: int, /, **kwargs: Any): ...

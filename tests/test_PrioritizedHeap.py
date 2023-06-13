import numpy as np
from typing import NamedTuple

from ReplayTables.PrioritizedHeap import PrioritizedHeap

class Data(NamedTuple):
    a: float | np.ndarray
    b: int | np.ndarray

class TestPrioritizedHeap:
    def test_simple_buffer(self):
        rng = np.random.default_rng(0)

        buffer = PrioritizedHeap(
            max_size=10,
            structure=Data,
            rng=rng,
        )

        # low priority items are not added
        buffer.add(Data(1, 2), priority=0.1)
        assert buffer.size() == 0

        # high priority items are added
        buffer.add(Data(2, 3), priority=2)
        assert buffer.size() == 1

        # items can be popped
        item = buffer.pop()
        assert item == Data(2, 3)
        assert buffer.size() == 0

        # many items can be added
        for i in range(30):
            buffer.add(Data(i, 2 * i), priority=(i - 14) ** 2)

        assert buffer.size() == 10

        item = buffer.pop()
        assert item == Data(29, 58)
        assert buffer.size() == 9

        # can sample a batch from the buffer
        batch, _, _ = buffer.sample(5)
        assert np.asarray(batch.a).size == 5
        assert np.all(batch.a == np.array([28, 0, 1, 27, 26]))
        assert buffer.size() == 4

# ----------------
# -- Benchmarks --
# ----------------

class TestBenchmarks:
    def test_prioritized_heap_add(self, benchmark):
        rng = np.random.default_rng(0)
        buffer = PrioritizedHeap(
            max_size=100_000,
            structure=Data,
            rng=rng,
        )
        d = Data(0.1, 1)

        def _inner(buffer, d):
            buffer.add(d, priority=2)

        benchmark(_inner, buffer, d)

    def test_prioritized_heap_sample(self, benchmark):
        rng = np.random.default_rng(0)
        buffer = PrioritizedHeap(
            max_size=100_000,
            structure=Data,
            rng=rng,
        )
        d = Data(0.1, 1)

        for i in range(100_000):
            buffer.add(d, priority=i)

        def _inner(buffer):
            buffer.sample(32)

        benchmark(_inner, buffer)

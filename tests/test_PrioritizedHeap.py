import unittest
import numpy as np
from typing import NamedTuple

from ReplayTables.PrioritizedHeap import PrioritizedHeap

class Data(NamedTuple):
    a: float | np.ndarray
    b: int | np.ndarray


class TestPrioritizedHeap(unittest.TestCase):
    def test_simple_buffer(self):
        rng = np.random.RandomState(0)

        buffer = PrioritizedHeap(
            max_size=10,
            structure=Data,
            rng=rng,
        )

        # low priority items are not added
        buffer.add(Data(1, 2), priority=0.1)
        self.assertEqual(buffer.size(), 0)

        # high priority items are added
        buffer.add(Data(2, 3), priority=2)
        self.assertEqual(buffer.size(), 1)

        # items can be popped
        item = buffer.pop()
        self.assertEqual(item, Data(2, 3))
        self.assertEqual(buffer.size(), 0)

        # many items can be added
        for i in range(30):
            buffer.add(Data(i, 2 * i), priority=(i - 14) ** 2)

        self.assertEqual(buffer.size(), 10)

        item = buffer.pop()
        self.assertEqual(item, Data(29, 58))
        self.assertEqual(buffer.size(), 9)

        # can sample a batch from the buffer
        batch, _, _ = buffer.sample(5)
        self.assertEqual(np.asarray(batch.a).size, 5)
        self.assertTrue(np.all(batch.a == np.array([28, 0, 1, 27, 26])))
        self.assertEqual(buffer.size(), 4)

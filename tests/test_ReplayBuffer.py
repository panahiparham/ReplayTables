import unittest
import numpy as np
import pickle
from typing import NamedTuple

from ReplayTables.ReplayBuffer import ReplayBuffer

class Data(NamedTuple):
    a: float | np.ndarray
    b: int | np.ndarray


class TestReplayBuffer(unittest.TestCase):
    def test_simple_buffer(self):
        rng = np.random.RandomState(0)
        buffer = ReplayBuffer(5, Data, rng)

        # on creation, the buffer should have no size
        self.assertEqual(buffer.size(), 0)

        # should be able to simply add and sample a single data point
        d = Data(a=0.1, b=1)
        buffer.add(d)
        self.assertEqual(buffer.size(), 1)
        samples, idxs, weights = buffer.sample(10)
        self.assertTrue(np.all(samples.b == 1))
        self.assertTrue(np.all(idxs == 0))
        self.assertTrue(np.all(weights == 1))

        # should be able to add a few more points
        for i in range(4):
            x = i + 2
            buffer.add(Data(a=x / 10, b=x))

        self.assertEqual(buffer.size(), 5)
        samples, idxs, weights = buffer.sample(1000)

        unique = np.unique(samples.b)
        unique.sort()

        self.assertTrue(np.all(unique == np.array([1, 2, 3, 4, 5])))

        # buffer drops the oldest element when over max size
        buffer.add(Data(a=0.6, b=6))
        self.assertEqual(buffer.size(), 5)

        samples, _, _ = buffer.sample(1000)
        unique = np.unique(samples.b)
        unique.sort()
        self.assertTrue(np.all(unique == np.array([2, 3, 4, 5, 6])))

    def test_getitem(self):
        rng = np.random.RandomState(0)
        buffer = ReplayBuffer(10, Data, rng)

        for i in range(15):
            buffer.add(Data(a=i, b=2 * i))

        # should be the most recently added item
        got, _, _ = buffer[0]
        expect = Data(a=np.array([14]), b=np.array([28]))
        self.assertEqual(got, expect)

        # should be oldest item in buffer
        got, _, _ = buffer[-1]
        expect = Data(a=np.array([5]), b=np.array([10]))
        self.assertEqual(got, expect)

        got, _, _ = buffer[2:7:2]
        expect = Data(
            a=np.array([12, 10, 8]),
            b=np.array([24, 20, 16]),
        )
        self.assertTrue(np.all(got.a == expect.a))
        self.assertTrue(np.all(got.b == expect.b))

    def test_pickleable(self):
        rng = np.random.RandomState(0)
        buffer = ReplayBuffer(5, Data, rng)

        for i in range(8):
            buffer.add(Data(a=i, b=i))

        byt = pickle.dumps(buffer)
        buffer2 = pickle.loads(byt)

        s, _, _ = buffer.sample(3)
        s2, _, _ = buffer2.sample(3)

        self.assertTrue(np.all(s.a == s2.a) and np.all(s.b == s2.b))

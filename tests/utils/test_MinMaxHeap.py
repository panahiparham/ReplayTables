import unittest
from ReplayTables._utils.MinMaxHeap import MinMaxHeap

class TestMinMaxHeap(unittest.TestCase):
    def test_can_track_stats(self):
        h = MinMaxHeap()

        h.add(1, 'a')
        p, got = h.min()
        self.assertEqual(p, 1)
        self.assertEqual(got, 'a')

        p, got = h.max()
        self.assertEqual(p, 1)
        self.assertEqual(got, 'a')

        h.add(3, 'b')
        p, got = h.min()
        self.assertEqual(p, 1)
        self.assertEqual(got, 'a')

        p, got = h.max()
        self.assertEqual(p, 3)
        self.assertEqual(got, 'b')

        for i in range(33):
            h.add(i, f'{i}')

        p, got = h.min()
        self.assertEqual(p, 0)
        self.assertEqual(got, '0')

        p, got = h.max()
        self.assertEqual(p, 32)
        self.assertEqual(got, '32')

        p, got = h.pop_min()
        self.assertEqual(p, 0)
        self.assertEqual(got, '0')

        p, got = h.pop_min()
        self.assertEqual(p, 1)
        self.assertEqual(got, '1')

        p, got = h.pop_max()
        self.assertEqual(p, 32)
        self.assertEqual(got, '32')

        p, got = h.pop_max()
        self.assertEqual(p, 31)
        self.assertEqual(got, '31')

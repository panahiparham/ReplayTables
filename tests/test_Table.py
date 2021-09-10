import unittest
import numpy as np

from ReplayTables.Table import Table

class TestTable(unittest.TestCase):
    def test_canAddMultipleColumns(self):
        table = Table(3, seed=0, columns=[
            { 'name': 'A', 'shape': 3, 'dtype': 'float64' },
            { 'name': 'B', 'shape': 1, 'dtype': bool },
            { 'name': 'C', 'shape': tuple(), 'dtype': 'int32' },
        ])

        table.addTuple((np.arange(3) * 1.1, False, 1))
        table.addTuple((np.arange(3) * 2.2, False, 2))
        table.addTuple((np.arange(3) * 3.3, True, 3))
        table.addTuple((np.arange(3) * 4.4, True, 4))
        table.addTuple((np.arange(3) * 5.5, False, 5))

        A, B, C = table.getAll()

        self.assertTrue(np.allclose(A, np.array([
            [0, 3.3, 6.6],
            [0, 4.4, 8.8],
            [0, 5.5, 11],
        ])))

        self.assertTrue(np.all(B == [[True], [True], [False]]))

        self.assertTrue(np.allclose(C, [3, 4, 5]))

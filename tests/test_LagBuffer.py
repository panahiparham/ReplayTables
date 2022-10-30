import unittest
from typing import Any, Optional
from dataclasses import dataclass

from ReplayTables.LagBuffer import LagBuffer, LaggedExperience

@dataclass
class Experience:
    s: Any
    a: int
    r: Optional[float]
    gamma: float
    terminal: bool


class TestLagBuffer(unittest.TestCase):
    def test_1_step(self):
        buffer = LagBuffer(1)

        exp1 = Experience(
            s=3,
            a=0,
            r=None,
            gamma=0.99,
            terminal=False,
        )

        exp2 = Experience(
            s=4,
            a=2,
            r=2.0,
            gamma=0,
            terminal=True,
        )

        out = list(buffer.add(exp1))
        self.assertEqual(out, [])

        out = list(buffer.add(exp2))
        self.assertEqual(len(out), 1)

        self.assertEqual(out[0], LaggedExperience(
            s=3,
            a=0,
            r=2.,
            gamma=0,
            terminal=True,
            sp=4,
        ))

    def test_3_step(self):
        buffer = LagBuffer(3)

        for i in range(3):
            exp = Experience(
                s=1 + i,
                a=i,
                r=2.0 * (i + 1),
                gamma=0.9,
                terminal=False,
            )
            out = list(buffer.add(exp))
            self.assertEqual(out, [])

        exp = Experience(
            s=4,
            a=22,
            r=8.0,
            gamma=0.9,
            terminal=False,
        )
        out = list(buffer.add(exp))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0], LaggedExperience(
            s=1,
            a=0,
            r=4 + (0.9 * 6) + (0.9 ** 2 * 8),
            gamma=0.9 ** 3,
            terminal=False,
            sp=4,
        ))

        term = Experience(
            s=5,
            a=33,
            r=10.0,
            gamma=0,
            terminal=True,
        )

        out = list(buffer.add(term))
        self.assertEqual(len(out), 3)

        self.assertEqual(out[0], LaggedExperience(
            s=2,
            a=1,
            r=6 + (0.9 * 8) + (0.9 ** 2 * 10),
            gamma=0.,
            terminal=True,
            sp=5
        ))

        self.assertEqual(out[1], LaggedExperience(
            s=3,
            a=2,
            r=8 + (0.9 * 10),
            gamma=0,
            terminal=True,
            sp=5,
        ))

        self.assertEqual(out[2], LaggedExperience(
            s=4,
            a=22,
            r=10,
            gamma=0,
            terminal=True,
            sp=5,
        ))

    def test_flush(self):
        buffer = LagBuffer(lag=1)

        buffer.add(Experience(
            s=0,
            a=0,
            r=0,
            gamma=0,
            terminal=True,
        ))

        self.assertEqual(len(buffer._buffer), 1)
        buffer.flush()
        self.assertEqual(len(buffer._buffer), 0)

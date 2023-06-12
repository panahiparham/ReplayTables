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


class TestLagBuffer:
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
        assert out == []

        out = list(buffer.add(exp2))
        assert len(out) == 1
        assert out[0] == LaggedExperience(
            s=3,
            a=0,
            r=2.,
            gamma=0,
            terminal=True,
            sp=4,
            raw=exp1,
        )

    def test_3_step(self):
        buffer = LagBuffer(3)

        experiences = []
        for i in range(3):
            exp = Experience(
                s=1 + i,
                a=i,
                r=2.0 * (i + 1),
                gamma=0.9,
                terminal=False,
            )
            out = list(buffer.add(exp))
            assert out == []

            experiences.append(exp)

        exp = Experience(
            s=4,
            a=22,
            r=8.0,
            gamma=0.9,
            terminal=False,
        )
        out = list(buffer.add(exp))
        assert len(out) == 1
        assert out[0] == LaggedExperience(
            s=1,
            a=0,
            r=4 + (0.9 * 6) + (0.9 ** 2 * 8),
            gamma=0.9 ** 3,
            terminal=False,
            sp=4,
            raw=experiences[0],
        )

        term = Experience(
            s=5,
            a=33,
            r=10.0,
            gamma=0,
            terminal=True,
        )

        out = list(buffer.add(term))
        assert len(out) == 3

        assert out[0] == LaggedExperience(
            s=2,
            a=1,
            r=6 + (0.9 * 8) + (0.9 ** 2 * 10),
            gamma=0.,
            terminal=True,
            sp=5,
            raw=experiences[1],
        )

        assert out[1] == LaggedExperience(
            s=3,
            a=2,
            r=8 + (0.9 * 10),
            gamma=0,
            terminal=True,
            sp=5,
            raw=experiences[2],
        )

        assert out[2] == LaggedExperience(
            s=4,
            a=22,
            r=10,
            gamma=0,
            terminal=True,
            sp=5,
            raw=exp,
        )

    def test_flush(self):
        buffer = LagBuffer(lag=1)

        buffer.add(Experience(
            s=0,
            a=0,
            r=0,
            gamma=0,
            terminal=True,
        ))

        assert len(buffer._buffer) == 1
        buffer.flush()
        assert len(buffer._buffer) == 0

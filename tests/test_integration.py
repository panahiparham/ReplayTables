import numpy as np
from typing import Any, NamedTuple

from ReplayTables.ReplayBuffer import ReplayBuffer
from ReplayTables.LagBuffer import LagBuffer

class Experience(NamedTuple):
    s: Any
    a: int
    r: float | None
    gamma: float
    terminal: bool

class Lagged(NamedTuple):
    s: Any
    a: Any
    r: Any
    gamma: Any
    sp: Any

# ----------------
# -- Benchmarks --
# ----------------
class TestBenchmarks:
    def test_1_step_loop(self, benchmark):
        def rl_loop(lag: LagBuffer, buffer: ReplayBuffer, d):
            for _ in range(100):
                for exp in lag.add(d):
                    l = Lagged(
                        s=exp.s,
                        a=exp.a,
                        r=exp.r,
                        gamma=exp.gamma,
                        sp=exp.sp,
                    )
                    buffer.add(l)
                    _ = buffer.sample(32)

        rng = np.random.default_rng(0)
        lag = LagBuffer(1)
        buffer = ReplayBuffer(30, Lagged, rng)
        d = Experience(
            s=np.zeros(50),
            a=0,
            r=0.1,
            gamma=0.99,
            terminal=False,
        )

        benchmark(rl_loop, lag, buffer, d)

    def test_3_step_loop(self, benchmark):
        def rl_loop(lag: LagBuffer, buffer: ReplayBuffer, d):
            for _ in range(100):
                for exp in lag.add(d):
                    l = Lagged(
                        s=exp.s,
                        a=exp.a,
                        r=exp.r,
                        gamma=exp.gamma,
                        sp=exp.sp,
                    )
                    buffer.add(l)
                    _ = buffer.sample(32)

        rng = np.random.default_rng(0)
        lag = LagBuffer(3)
        buffer = ReplayBuffer(30, Lagged, rng)
        d = Experience(
            s=np.zeros(50),
            a=0,
            r=0.1,
            gamma=0.99,
            terminal=False,
        )

        benchmark(rl_loop, lag, buffer, d)

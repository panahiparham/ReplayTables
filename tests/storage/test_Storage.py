import pytest
import numpy as np

from collections import deque
from typing import cast, Any, Type
from ReplayTables.storage.BasicStorage import Storage, BasicStorage
from ReplayTables.storage.CompressedStorage import CompressedStorage
from ReplayTables.storage.NonArrayStorage import NonArrayStorage
from ReplayTables.interface import EIDs, LaggedTimestep

from tests._utils.fake_data import fake_lagged_timestep, LaggedDataStream, lagged_equal, batch_equal, lags_to_batch


STORAGES = [
    BasicStorage,
    CompressedStorage,
    NonArrayStorage,
]

BATCH_STORAGES = [
    BasicStorage,
    CompressedStorage,
]

@pytest.mark.parametrize('Store', STORAGES)
def test_add1(Store: Type[Storage]):
    storage = Store(10)
    storage.add(
        fake_lagged_timestep(eid=32, xid=32, n_xid=34),
    )

    storage.add(
        fake_lagged_timestep(eid=34, xid=34, n_xid=36),
    )

    assert len(storage) == 2

    d = storage.get(cast(EIDs, np.array([34])))
    assert d.eid == 34

    for i in range(10):
        storage.add(
            fake_lagged_timestep(eid=36 + i, xid=36 + i, n_xid=38 + i),
        )

    assert len(storage) == 10


@pytest.mark.parametrize('Store', STORAGES)
def test_integration1(Store: Type[Storage]):
    storage = Store(10)
    data = LaggedDataStream(lag=1)
    past = deque(maxlen=5)

    def add_and_check(exp: LaggedTimestep, expected_len: int):
        past.append(exp)
        storage.add(exp)
        assert len(storage) == expected_len

        got = storage.get_item(exp.eid)
        assert lagged_equal(got, exp)

        for past_exp in past:
            got = storage.get_item(past_exp.eid)
            assert lagged_equal(got, past_exp)

    # can maintain partial storage
    # ----------------------------
    exps = data.next()
    assert len(exps) == 0
    assert len(storage) == 0

    exps = data.next()
    assert len(exps) == 1

    add_and_check(exps[0], expected_len=1)

    expected_length = 1
    for _ in range(100):
        expected_length = min(expected_length + 1, 10)

        exp = data.next_single()
        add_and_check(exp, expected_length)

    # make sure soft-termination works as expected
    # --------------------------------------------
    exps = data.next(soft_term=True)
    assert len(exps) == 1

    add_and_check(exps[0], expected_len=10)

    # first experience after soft-term should be null
    # since lag buffer should be reset
    exps = data.next()
    assert len(exps) == 0

    for _ in range(15):
        exp = data.next_single()
        add_and_check(exp, expected_len=10)

    # check hard termination
    # ----------------------
    exps = data.next(hard_term=True)
    assert len(exps) == 1
    assert exps[0].n_xid is None
    assert exps[0].n_x is None

    add_and_check(exps[0], expected_len=10)

    exps = data.next()
    assert len(exps) == 0

    for _ in range(15):
        exp = data.next_single()
        add_and_check(exp, expected_len=10)


@pytest.mark.parametrize('Store', BATCH_STORAGES)
def test_integration2(Store: Type[Storage]):
    storage = Store(10)
    data = LaggedDataStream(lag=1)
    data.next()

    samples = []
    for _ in range(10):
        d = data.next_single()
        storage.add(d)
        samples.append(d)

    assert len(storage) == 10

    eids: Any = np.arange(10)
    got = storage.get(eids)

    expected = lags_to_batch(samples)
    assert batch_equal(got, expected)

    # can handle soft termination
    # ---------------------------
    samples = []
    exps = data.next(soft_term=True)
    samples.append(exps[0])
    storage.add(exps[0])

    data.next()

    for _ in range(5):
        d = data.next_single()
        samples.append(d)
        storage.add(d)

    expected = lags_to_batch(samples)
    eids = expected.eid
    got = storage.get(eids)

    assert batch_equal(got, expected)
    assert np.all(got.xp[0] != 0)
    assert got.terminal[0]

    # can handle hard termination
    # ---------------------------
    samples = []
    exps = data.next(hard_term=True)
    samples.append(exps[0])
    storage.add(exps[0])

    data.next()

    for _ in range(5):
        d = data.next_single()
        samples.append(d)
        storage.add(d)

    expected = lags_to_batch(samples)
    eids = expected.eid
    got = storage.get(eids)

    assert batch_equal(got, expected)
    assert np.all(got.xp[0] == 0)
    assert got.terminal[0]

# ------------------------------
# -- Performance Benchmarking --
# ------------------------------

@pytest.mark.parametrize('Store', STORAGES)
def test_small_data(benchmark, Store: Type[Storage]):
    benchmark.name = Store.__name__
    benchmark.group = 'storage | small data'

    def add_and_get(storage: Storage, timesteps, eids):
        for i in range(100):
            storage.add(timesteps[i])

        for i in range(100):
            storage.get(eids)

    storage = Store(10_000)
    eids = np.arange(32)
    data = [
        fake_lagged_timestep(eid=2 * i, xid=2 * i, n_xid=2 * i + 1, x=np.ones(10), n_x=np.ones(10))
        for i in range(100)
    ]

    benchmark(add_and_get, storage, data, eids)


@pytest.mark.parametrize('Store', STORAGES)
def test_big_data(benchmark, Store: Type[Storage]):
    benchmark.name = Store.__name__
    benchmark.group = 'storage | big data'

    def add_and_get(storage: Storage, timesteps, eids):
        for i in range(100):
            storage.add(timesteps[i])

        for i in range(100):
            storage.get(eids)

    storage = Store(10_000)
    eids = np.arange(32)
    x = np.ones((64, 64, 3), dtype=np.uint8)
    data = [
        fake_lagged_timestep(eid=2 * i, xid=2 * i, n_xid=2 * i + 1, x=x, n_x=x)
        for i in range(100)
    ]

    benchmark(add_and_get, storage, data, eids)

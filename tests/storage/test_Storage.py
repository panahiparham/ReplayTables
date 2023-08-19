import pytest
import numpy as np

from typing import cast, Any, Type
from ReplayTables.storage.BasicStorage import Storage, BasicStorage
from ReplayTables.storage.CompressedStorage import CompressedStorage
from ReplayTables.storage.NonArrayStorage import NonArrayStorage
from ReplayTables.interface import IDX, IDXs

from tests._utils.fake_data import fake_lagged_timestep


STORAGES = [
    BasicStorage,
    CompressedStorage,
    NonArrayStorage,
]

@pytest.mark.parametrize('Store', STORAGES)
def test_add1(Store: Type[Storage]):
    storage = Store(10)
    storage.add(
        cast(IDX, 0),
        cast(IDX, 1),
        fake_lagged_timestep(eid=32, n_eid=34),
    )

    storage.add(
        cast(IDX, 1),
        cast(IDX, 2),
        fake_lagged_timestep(eid=34, n_eid=36),
    )

    assert len(storage) == 2

    d = storage.get(cast(IDXs, np.array([1])))
    assert d.eid == 34

    for i in range(10):
        storage.add(
            cast(IDX, (3 + i) % 10),
            cast(IDX, (4 + i) % 10),
            fake_lagged_timestep(eid=36 + i, n_eid=38 + i),
        )

    assert len(storage) == 10


@pytest.mark.parametrize('Store', STORAGES)
def test_small_data(benchmark, Store: Type[Storage]):
    benchmark.name = Store.__name__
    benchmark.group = 'storage | small data'

    def add_and_get(storage: Storage, timestep, idxs):
        for i in range(100):
            idx: Any = i
            n_idx: Any = 100 + i
            storage.add(idx, n_idx, timestep)

        for i in range(100):
            storage.get(idxs)

    storage = Store(10_000)
    idxs = np.arange(32)
    d = fake_lagged_timestep(eid=0, n_eid=1, x=np.ones(10), n_x=np.ones(10))

    benchmark(add_and_get, storage, d, idxs)


@pytest.mark.parametrize('Store', STORAGES)
def test_big_data(benchmark, Store: Type[Storage]):
    benchmark.name = Store.__name__
    benchmark.group = 'storage | big data'

    def add_and_get(storage: Storage, timestep, idxs):
        for i in range(100):
            idx: Any = i
            n_idx: Any = 100 + i
            storage.add(idx, n_idx, timestep)

        for i in range(100):
            storage.get(idxs)

    storage = Store(10_000)
    idxs = np.arange(32)
    x = np.ones((64, 64, 3), dtype=np.uint8)
    d = fake_lagged_timestep(eid=0, n_eid=1, x=x, n_x=x)

    benchmark(add_and_get, storage, d, idxs)

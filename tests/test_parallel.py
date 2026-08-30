"""
Tests for espei.parallel, ESPEI's parallelization backends.
"""

import logging
import multiprocessing
import subprocess
import sys
import time

import numpy as np
import pytest
from pycalphad import Database

import espei.parallel
from espei.optimizers.opt_mcmc import EmceeOptimizer
from espei.parallel import DaskPool, MultiprocessingPool, make_scheduler, _raise_dask_work_stealing

from .fixtures import datasets_db
from .testing_data import (
    CU_MG_TDB,
    CU_MG_DATASET_ZPF_ZERO_ERROR,
    CU_MG_EXP_ACTIVITY,
    CU_MG_CPM_MIX_X_HCP_A3,
    CU_MG_SM_MIX_T_X_FCC_A1,
    CU_MG_EQ_HMR_LIQUID,
)


def identity(x):
    return x


def square(x):
    return x * x


def staggered_square(x):
    """Square ``x``, taking longer for smaller ``x``.

    Items therefore finish out of order, so a ``map`` that returned completion
    order rather than input order would fail the ordering assertion. The unit is
    only as long as it needs to be to invert the order reliably; the sleeps are
    otherwise pure test runtime.
    """
    time.sleep(0.002 * (10 - x))
    return x * x


class SerialPool:
    """Trivial reference pool: builtin ``map``, no parallelism."""

    def map(self, f, iterable):
        return list(map(f, iterable))

    def close(self):
        pass


try:
    import distributed  # noqa: F401
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask and distributed are not installed")


def _serial_pool():
    return SerialPool()


def _dask_pool():
    return DaskPool(cores=2)


# Parameterization shared by every conformance test. MultiprocessingPool is not
# here: starting its workers costs about a second and a half, almost all of it
# importing espei, and test_multiprocessing_pool_mcmc_matches_serial already
# exercises its map against a real process boundary. Everything else about it is
# bookkeeping in this process, tested against a stand-in executor below.
POOL_FACTORIES = [
    pytest.param(_serial_pool, id="serial"),
    pytest.param(_dask_pool, id="dask", marks=requires_dask),
]


@pytest.fixture(scope="module")
def pool(request):
    """Yield a pool built by the requested factory and close it afterwards.

    Module scoped, so each backend starts its workers once for all of the
    conformance tests. Those tests must therefore all map the same callable:
    mapping a different one re-pins it, which rebuilds the workers.
    """
    pl = request.param()
    try:
        yield pl
    finally:
        pl.close()




@pytest.mark.parametrize("pool", POOL_FACTORIES, indirect=True)
def test_pool_map_returns_materialized_results_in_input_order(pool):
    """pool.map must block and return concrete results in the order of the input."""
    values = list(range(10))
    result = pool.map(staggered_square, values)
    # Blocking: the return value is already a concrete sequence, not futures or
    # a lazy iterator. Indexing/len must work without any further waiting.
    assert len(result) == len(values)
    # Order-preserving, despite the staggered runtimes.
    assert list(result) == [square(x) for x in values]


@pytest.mark.parametrize("pool", POOL_FACTORIES, indirect=True)
def test_pool_map_is_reusable_across_calls(pool):
    """Successive map calls on the same pool must each return correct results."""
    for values in ([0, 1, 2, 3], [4, 5], [6, 7, 8, 9, 10]):
        assert list(pool.map(staggered_square, values)) == [square(x) for x in values]


@pytest.mark.parametrize("pool", POOL_FACTORIES, indirect=True)
def test_pool_map_handles_an_empty_iterable(pool):
    assert list(pool.map(staggered_square, [])) == []


class RecordingExecutor:
    """Stands in for ProcessPoolExecutor, without the worker processes.

    Real workers cost about a second and a half each to start, nearly all of it
    importing espei. What MultiprocessingPool does around the executor -- when
    it builds one, what it pins, when it rebuilds -- is decided in this process,
    so it can be checked against a stand-in. That the arrangement actually works
    across a process boundary is test_multiprocessing_pool_mcmc_matches_serial.
    """

    def __init__(self, max_workers, mp_context=None, initializer=None, initargs=()):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.initargs = initargs
        self.maps = []
        self.shutdowns = []
        RecordingExecutor.built.append(self)
        # A real pool runs the initializer once per worker, at startup, and
        # never again -- which is the whole point of pinning through it.
        initializer(*initargs)

    def map(self, fn, args, chunksize=None):
        self.maps.append((fn, list(args), chunksize))
        return iter([fn(a) for a in args])  # lazy, as the real map is

    def shutdown(self, cancel_futures=False):
        self.shutdowns.append(cancel_futures)


@pytest.fixture
def recording_executor(monkeypatch):
    """Build MultiprocessingPool against RecordingExecutor instead of a real one."""
    monkeypatch.setattr(espei.parallel, "ProcessPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(espei.parallel, "_PINNED_FN", None)
    monkeypatch.setattr(RecordingExecutor, "built", [], raising=False)
    return RecordingExecutor.built


def test_multiprocessing_pool_pins_the_mapped_callable_once(recording_executor):
    """The callable is shipped once, at worker startup, not once per map call.

    This is the #230 optimization: the callable emcee passes carries the whole
    MCMC context, so shipping it per call rather than per worker would put the
    context on the wire for every walker of every iteration.
    """
    pool = MultiprocessingPool(2)
    assert recording_executor == [], "workers must not start before the first map"

    assert pool.map(square, [1, 2, 3]) == [1, 4, 9]
    assert pool.map(square, [4, 5]) == [16, 25]
    assert pool.map(square, []) == []

    # One executor for all three calls, built with square pinned through the
    # initializer, and mapping the module-level trampoline rather than square
    # itself -- so what crosses the boundary per task is a walker position.
    assert len(recording_executor) == 1
    executor = recording_executor[0]
    assert executor.max_workers == 2
    assert executor.initargs == (square,)
    assert [fn for fn, _args, _chunk in executor.maps] == [espei.parallel._call_installed] * 3
    assert [chunk for _fn, _args, chunk in executor.maps] == [1, 1, 1]


def test_multiprocessing_pool_repins_when_the_mapped_callable_changes(recording_executor):
    """Mapping a different callable re-pins it rather than silently reusing the old one.

    The failure mode is quiet and wrong: every result computed by the previously
    pinned callable.
    """
    pool = MultiprocessingPool(2)
    assert pool.map(square, [1, 2, 3]) == [1, 4, 9]
    assert pool.map(identity, [1, 2, 3]) == [1, 2, 3]

    assert len(recording_executor) == 2, "the workers must be rebuilt to re-pin"
    assert recording_executor[1].initargs == (identity,)
    # The superseded executor is shut down rather than left holding workers.
    assert recording_executor[0].shutdowns == [True]


def test_multiprocessing_pool_map_blocks_and_close_is_idempotent(recording_executor):
    """map returns a materialized list; close tears down once and only once."""
    pool = MultiprocessingPool(2)
    pool.close()  # never mapped: must not raise, and must not build anything
    assert recording_executor == []

    result = pool.map(square, [1, 2, 3])
    assert isinstance(result, list), "map must block, not hand back an iterator"

    pool.close()
    pool.close()
    assert recording_executor[0].shutdowns == [True], "close must not shut down twice"
    assert pool._pool is None


def test_make_scheduler_multiprocessing_uses_every_core_by_default():
    pool = make_scheduler({'scheduler': 'multiprocessing'})
    assert isinstance(pool, MultiprocessingPool)
    assert pool.cores == multiprocessing.cpu_count()


def test_make_scheduler_honors_cores():
    assert make_scheduler({'scheduler': 'multiprocessing', 'cores': 2}).cores == 2


def test_make_scheduler_caps_cores_at_the_available_cores(caplog):
    """Cores is validated eagerly here, because workers start at the first map."""
    available = multiprocessing.cpu_count()
    with caplog.at_level(logging.WARNING, logger="espei.parallel"):
        pool = make_scheduler({'scheduler': 'multiprocessing', 'cores': available + 1})
    assert pool.cores == available
    assert "larger than available" in caplog.text


def test_make_scheduler_returns_none_for_a_null_scheduler(caplog):
    with caplog.at_level(logging.WARNING, logger="espei.parallel"):
        assert make_scheduler({'scheduler': None}) is None
    assert caplog.text == ""


def test_make_scheduler_warns_that_cores_is_ignored_without_a_scheduler(caplog):
    """Cores sizes a pool ESPEI starts, so it means nothing for a serial run."""
    with caplog.at_level(logging.WARNING, logger="espei.parallel"):
        assert make_scheduler({'scheduler': None, 'cores': 2}) is None
    assert "'cores' setting has no effect" in caplog.text


def test_make_scheduler_passes_a_json_path_as_a_dask_scheduler_file(monkeypatch, caplog):
    """A *.json scheduler connects to an externally managed cluster, which sizes itself."""
    calls = []
    monkeypatch.setattr(espei.parallel, "DaskPool", lambda **kwargs: calls.append(kwargs))
    settings = {'scheduler': 'my-scheduler.json', 'cores': 2}
    with caplog.at_level(logging.WARNING, logger="espei.parallel"):
        make_scheduler(settings, log_verbosity=2, log_filename='espei.log')
    assert calls == [dict(scheduler_file='my-scheduler.json', log_verbosity=2, log_filename='espei.log')]
    assert "'cores' setting has no effect" in caplog.text


def test_make_scheduler_builds_a_dask_pool_with_a_local_cluster(monkeypatch, caplog):
    """The 'dask' setting sizes a cluster of its own, unlike a scheduler file.

    Building a real one costs a second and a half of cluster startup to learn
    what these two lines already say; the conformance tests above cover whether
    a DaskPool works.
    """
    calls = []
    monkeypatch.setattr(espei.parallel, "DaskPool", lambda **kwargs: calls.append(kwargs))
    with caplog.at_level(logging.WARNING, logger="espei.parallel"):
        make_scheduler({'scheduler': 'dask', 'cores': 2}, log_verbosity=2, log_filename='espei.log')
    assert calls == [dict(cores=2, log_verbosity=2, log_filename='espei.log')]
    assert caplog.text == "", "cores sizes this pool, so it must not warn that it is ignored"


@requires_dask
def test_raise_dask_work_stealing():
    """Work stealing destabilizes long runs, so ESPEI refuses to run with it on."""
    import dask
    with dask.config.set({"distributed.scheduler.work-stealing": False}):
        _raise_dask_work_stealing()  # must not raise
    with dask.config.set({"distributed.scheduler.work-stealing": True}):
        with pytest.raises(ValueError):
            _raise_dask_work_stealing()


def _insert_all_residual_type_datasets(datasets_db):
    """Insert one dataset for each of the four registered residual types."""
    datasets_db.insert(CU_MG_DATASET_ZPF_ZERO_ERROR)  # ZPFResidual
    datasets_db.insert(CU_MG_EXP_ACTIVITY)  # ActivityResidual
    datasets_db.insert(CU_MG_CPM_MIX_X_HCP_A3)  # FixedConfigurationPropertyResidual
    datasets_db.insert(CU_MG_SM_MIX_T_X_FCC_A1)  # FixedConfigurationPropertyResidual
    datasets_db.insert(CU_MG_EQ_HMR_LIQUID)  # EquilibriumPropertyResidual


def test_multiprocessing_pool_mcmc_matches_serial(datasets_db):
    """An MCMC run through a MultiprocessingPool reproduces the serial run exactly.

    Covers all four residual types in a single run, so the whole context (deep
    copied Database, PickleableTinyDB datasets, eagerly built pycalphad
    PhaseRecordFactory objects wrapping symengine-compiled functions) crosses a
    real process boundary with no ESPEI-side serialization shims.
    """
    _insert_all_residual_type_datasets(datasets_db)
    symbols = ["VV0000", "VV0001"]
    fit_kwargs = dict(iterations=1, chains_per_parameter=2, deterministic=True)

    # A fresh Database for each optimizer: pycalphad's Database.__deepcopy__
    # shares the ``symbols`` dict, so ``fit`` writes its result back into the
    # Database it was given.
    serial_opt = EmceeOptimizer(Database(CU_MG_TDB))
    serial_opt.fit(symbols, datasets_db, **fit_kwargs)

    pool = MultiprocessingPool(2)
    try:
        parallel_opt = EmceeOptimizer(Database(CU_MG_TDB), scheduler=pool)
        parallel_opt.fit(symbols, datasets_db, **fit_kwargs)
    finally:
        pool.close()

    # The sampling decisions are identical: same proposals, same accept/reject.
    assert parallel_opt.sampler.chain.shape == serial_opt.sampler.chain.shape
    assert np.all(parallel_opt.sampler.chain == serial_opt.sampler.chain)
    # The log probabilities agree to floating point tolerance rather than
    # bit-for-bit. ActivityResidual and EquilibriumPropertyResidual run a
    # pycalphad equilibrium solve, which is not bit-reproducible even between
    # two runs in the same process (~1e-16 relative); the residual types that do
    # not solve an equilibrium do agree bit-for-bit across the process boundary.
    assert np.allclose(
        parallel_opt.sampler.lnprobability, serial_opt.sampler.lnprobability,
        rtol=1e-10, atol=0.0,
    )
    # Real numbers, not the -inf that a silently broken context would produce.
    assert np.all(np.isfinite(serial_opt.sampler.lnprobability))
    assert np.all(np.isfinite(parallel_opt.sampler.lnprobability))

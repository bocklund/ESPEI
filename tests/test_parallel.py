"""
Tests for espei.parallel, ESPEI's parallelization backends.
"""

import logging
import multiprocessing
import os
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


def _multiprocessing_pool():
    return MultiprocessingPool(2)


def _dask_pool():
    return DaskPool(cores=2)


# Parameterization shared by every conformance test.
POOL_FACTORIES = [
    pytest.param(_serial_pool, id="serial"),
    pytest.param(_multiprocessing_pool, id="multiprocessing"),
    pytest.param(_dask_pool, id="dask", marks=requires_dask),
]


@pytest.fixture
def pool(request):
    """Yield a pool built by the requested factory and close it afterwards."""
    pl = request.param()
    try:
        yield pl
    finally:
        pl.close()


# Module-level (picklable by reference) callables for the conformance tests.

def _identity(x):
    return x

def _square(x):
    return x * x


def _staggered_square(x):
    """Square ``x``, taking longer for smaller ``x``.

    Items therefore finish out of order, so a ``map`` that returned completion
    order rather than input order would fail the ordering assertion.
    """
    time.sleep(0.05 * (10 - x))
    return x * x



@pytest.mark.parametrize("pool", POOL_FACTORIES, indirect=True)
def test_pool_map_returns_materialized_results_in_input_order(pool):
    """pool.map must block and return concrete results in the order of the input."""
    values = list(range(10))
    result = pool.map(_staggered_square, values)
    # Blocking: the return value is already a concrete sequence, not futures or
    # a lazy iterator. Indexing/len must work without any further waiting.
    assert len(result) == len(values)
    # Order-preserving, despite the staggered runtimes.
    assert list(result) == [_square(x) for x in values]


@pytest.mark.parametrize("pool", POOL_FACTORIES, indirect=True)
def test_pool_map_is_reusable_across_calls(pool):
    """Successive map calls on the same pool must each return correct results."""
    for values in ([0, 1, 2, 3], [4, 5], [6, 7, 8, 9, 10]):
        assert list(pool.map(_square, values)) == [_square(x) for x in values]


@pytest.mark.parametrize("pool", POOL_FACTORIES, indirect=True)
def test_pool_map_handles_an_empty_iterable(pool):
    assert list(pool.map(_square, [])) == []


def test_multiprocessing_pool_repins_when_the_mapped_callable_changes():
    """Mapping a different callable re-pins it rather than silently reusing the old one."""
    pool = MultiprocessingPool(2)
    try:
        assert list(pool.map(_square, [1, 2, 3])) == [1, 4, 9]
        results = pool.map(_identity, [1, 2, 3])
        assert results == [1, 2, 3]
    finally:
        pool.close()


def test_multiprocessing_pool_close_is_a_noop_before_any_map():
    """Workers are created lazily, so close() on an unused pool must not fail."""
    pool = MultiprocessingPool(2)
    assert pool._pool is None
    pool.close()  # must not raise
    pool.close()  # idempotent
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


@requires_dask
def test_make_scheduler_builds_a_working_dask_pool():
    pool = make_scheduler({'scheduler': 'dask', 'cores': 1})
    try:
        assert isinstance(pool, DaskPool)
        assert list(pool.map(_square, [1, 2, 3])) == [1, 4, 9]
    finally:
        pool.close()


@requires_dask
def test_raise_dask_work_stealing():
    """Work stealing destabilizes long runs, so ESPEI refuses to run with it on."""
    import dask
    with dask.config.set({"distributed.scheduler.work-stealing": False}):
        _raise_dask_work_stealing()  # must not raise
    with dask.config.set({"distributed.scheduler.work-stealing": True}):
        with pytest.raises(ValueError):
            _raise_dask_work_stealing()


# Blocking dask has to happen in a subprocess: unimporting distributed from a
# session that already imported it is not something distributed supports.
# A None entry in sys.modules makes `import dask` raise ImportError while
# `importlib.util.find_spec("dask")` still politely reports "not installed",
# which other packages (e.g. pint) probe for at import time.
_BLOCK_DASK = """
import sys
sys.modules["dask"] = None
sys.modules["distributed"] = None
"""

_IMPORTS_WITHOUT_DASK = _BLOCK_DASK + """
import espei
import espei.utils
import espei.datasets
import espei.paramselect
import espei.parallel
import espei.espei_script
"""

_DASK_POOL_WITHOUT_DASK = _BLOCK_DASK + """
from espei.parallel import DaskPool
try:
    DaskPool()
except ImportError as exc:
    assert "espei[dask]" in str(exc), str(exc)
else:
    raise AssertionError("DaskPool did not raise without dask installed")
"""


_MAKE_SCHEDULER_WITHOUT_DASK = _BLOCK_DASK + """
from espei.parallel import make_scheduler
try:
    make_scheduler({'scheduler': 'dask'})
except ImportError as exc:
    assert "espei[dask]" in str(exc), str(exc)
else:
    raise AssertionError("make_scheduler did not raise without dask installed")
assert make_scheduler({'scheduler': 'multiprocessing'}) is not None
"""


def _run_snippet(source):
    return subprocess.run([sys.executable, "-c", source],
                          capture_output=True, text=True, timeout=300)


def test_espei_imports_without_dask():
    """The point of the dask extra: ESPEI must import with dask unavailable."""
    proc = _run_snippet(_IMPORTS_WITHOUT_DASK)
    assert proc.returncode == 0, proc.stderr


def test_dask_pool_without_dask_names_the_extra():
    proc = _run_snippet(_DASK_POOL_WITHOUT_DASK)
    assert proc.returncode == 0, proc.stderr


def test_make_scheduler_without_dask_names_the_extra():
    """Asking for dask without the extra must be actionable; the default still works."""
    proc = _run_snippet(_MAKE_SCHEDULER_WITHOUT_DASK)
    assert proc.returncode == 0, proc.stderr



_POOL_SCRIPT_BODY = """
from espei.parallel import MultiprocessingPool

def square(x):
    return x * x

def run():
    pool = MultiprocessingPool(2)
    try:
        print("RESULT", pool.map(square, range(4)))
    finally:
        pool.close()
"""

UNGUARDED_SCRIPT = _POOL_SCRIPT_BODY + "\nrun()\n"
GUARDED_SCRIPT = _POOL_SCRIPT_BODY + "\nif __name__ == '__main__':\n    run()\n"


def _run_script(tmp_path, name, source):
    script = tmp_path / name
    script.write_text(source)
    # The timeout is part of the assertion: a multiprocessing.Pool replaces
    # workers that die, so the unguarded case would not fail, it would run
    # forever.
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=300,
    )


def test_multiprocessing_pool_works_from_a_script_with_a_main_guard(tmp_path):
    proc = _run_script(tmp_path, "guarded.py", GUARDED_SCRIPT)
    assert proc.returncode == 0, proc.stderr
    assert "RESULT [0, 1, 4, 9]" in proc.stdout


def test_multiprocessing_pool_without_a_main_guard_fails_fast(tmp_path):
    """A missing __main__ guard must terminate with an error, not respawn forever."""
    proc = _run_script(tmp_path, "unguarded.py", UNGUARDED_SCRIPT)
    assert proc.returncode != 0
    # ProcessPoolExecutor reports the dead worker once and gives up. Under a
    # multiprocessing.Pool this repeats until the process is killed.
    assert proc.stderr.count("freeze_support") < 10
    assert "BrokenProcessPool" in proc.stderr



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

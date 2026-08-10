"""
Parallelization backends ("pools") for ESPEI.

The contract for a pool is exactly one method::

    pool.map(f, iterable) -> list

which must block until every result is available and must return the results in
the order of the input. That is the same contract ``emcee`` documents for the
object passed as ``pool=`` to :class:`emcee.EnsembleSampler`, so any object
satisfying it can be passed to
:class:`~espei.optimizers.opt_mcmc.EmceeOptimizer` as ``scheduler=``. Pools may
additionally offer ``close()``, which ESPEI calls when it can.

``emcee`` builds the mapped callable exactly once and passes *that same object*
to ``pool.map`` on every call, so pools in this module pin the callable by
object identity and ship it to the workers only once. This preserves the
optimization described in `ESPEI issue #230
<https://github.com/PhasesResearchLab/ESPEI/issues/230>`_ without any pool
needing to know anything about ``emcee``.
"""

import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from espei.logger import config_logger

_log = logging.getLogger(__name__)

__all__ = ["DaskPool", "MultiprocessingPool", "make_scheduler"]


# Worker-process state. ``_PINNED_FN`` is installed once per worker by
# ``_install_fn`` so the (large) callable and its bound context cross the
# process boundary once per worker instead of once per ``map`` call.
_PINNED_FN = None


def _install_fn(f):
    """
    Pin the mapped callable in this process. Runs once per worker at startup.

    Examples
    --------
    >>> _install_fn(abs)
    >>> _call_installed(-1.5)
    1.5

    """
    global _PINNED_FN
    _PINNED_FN = f


def _call_installed(x):
    """Call the pinned callable on ``x``. Picklable by reference."""
    return _PINNED_FN(x)


class MultiprocessingPool:
    """
    A pool of ``cores`` worker processes, backed by the standard library.

    Worker processes are started on the first call to :meth:`map`, because
    ``initializer=`` is the only mechanism the standard library offers for
    pinning state in workers and the callable to pin is not known until it is
    mapped. Mapping a different callable rebuilds the workers so that it is
    pinned instead; ``emcee`` passes the same callable every time, so this does
    not happen during a single MCMC run.

    Workers are started with the ``spawn`` start method, which requires that
    callers run ESPEI under an ``if __name__ == "__main__":`` guard. ``fork`` is
    unsafe in a process with threads running, which pycalphad's BLAS-threaded
    stack makes likely, and is unavailable on Windows.
    """

    def __init__(self, cores: int):
        self.cores = cores
        self._pool = None
        self._pinned_fn = None

    def map(self, f, iterable):
        """
        Return ``[f(x) for x in iterable]``, computed by the worker processes.

        Blocks until every result is available and preserves input order.
        """
        args = list(iterable)
        if f is not self._pinned_fn:
            self.close()
            _log.debug("Starting %s multiprocessing worker(s).", self.cores)
            self._pool = ProcessPoolExecutor(
                self.cores,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_install_fn,
                initargs=(f,),
            )
            self._pinned_fn = f
        # chunksize=1 because each call is expensive and roughly equal cost, so
        # the default chunking only hurts load balance. map() is lazy, so list()
        # is what blocks; it also preserves the input order.
        return list(self._pool.map(_call_installed, args, chunksize=1))

    def close(self):
        """Shut down the worker processes. Does nothing if they never started."""
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None
            self._pinned_fn = None


def _import_distributed():
    """Import dask and distributed, or raise pointing at the ``dask`` extra."""
    try:
        import dask
        import distributed
    except ImportError as exc:
        raise ImportError(
            "The 'dask' scheduler requires dask and distributed, which are not "
            "installed. Install them with `pip install espei[dask]`, or use the "
            "default 'multiprocessing' scheduler."
        ) from exc
    return dask, distributed


def _raise_dask_work_stealing():
    """
    Raise if work stealing is turned on in dask

    Raises
    -------
    ValueError
    """
    dask, _ = _import_distributed()
    has_work_stealing = dask.config.get('distributed.scheduler.work_stealing')
    if has_work_stealing:
        raise ValueError("The parameter 'distributed.scheduler.work-stealing' is on in dask. "
                         "This parameter causes some instability for long-running processes. "
                         "As of ESPEI v0.7.9, 'work-stealing' should be disabled automatically. "
                         "If you are seeing this error, please contact a developer.")


def _apply(x, fn):
    """Call the scattered callable ``fn`` on ``x``. Picklable by reference."""
    return fn(x)


class DaskPool:
    """
    A pool backed by a ``dask.distributed`` cluster.

    Starts a local cluster of ``cores`` single-threaded worker processes, unless
    an existing ``cluster`` (a cluster object or a scheduler address) or a
    ``scheduler_file`` written by an externally managed scheduler is given.
    ``dask`` and ``distributed`` are imported here rather than at module scope,
    so they are only needed by users who ask for this pool.

    Worker-side setup that has no equivalent in other pools lives here: ESPEI's
    logging configuration and NumPy print options are applied on every worker,
    and per-iteration worker memory is logged from :meth:`map`.
    """

    def __init__(self, cluster=None, scheduler_file=None, cores=None,
                 log_verbosity=0, log_filename=None):
        dask, distributed = _import_distributed()
        # Work stealing causes instability in long-running processes (ESPEI #134).
        dask.config.set({'distributed.scheduler.work-stealing': False})
        _raise_dask_work_stealing()
        if cluster is None and scheduler_file is None:
            # memory_limit=0 lets the system manage memory, so dask does not
            # pause or kill workers holding the (large) pinned context.
            cluster = distributed.LocalCluster(n_workers=cores, threads_per_worker=1, processes=True, memory_limit=0)
            self._owned_cluster = cluster
        else:
            self._owned_cluster = None
        self._client = distributed.Client(cluster, scheduler_file=scheduler_file)
        # The active memory manager removes data duplicated across workers, which
        # would leave the pinned callable on one worker and force every other
        # worker to fetch it over the network.
        self._client.amm.stop()
        self._client.run(config_logger, verbosity=log_verbosity, filename=log_filename)
        self._client.run(np.set_printoptions, linewidth=sys.maxsize)
        try:
            _log.info("dask dashboard at %s", self._client.dashboard_link)
        except KeyError:
            _log.info("Install bokeh to use the dask dashboard.")
        _log.info("Running with dask scheduler: %s [%s cores]", self._client.scheduler, sum(self._client.nthreads().values()))
        self._pinned_fn = None
        self._pinned_future = None
        self._num_maps = 0

    def map(self, f, iterable):
        """
        Return ``[f(x) for x in iterable]``, computed by the cluster's workers.

        Blocks until every result is available and preserves input order.
        """
        # Scattering pins the (large) callable on every worker so that it crosses
        # the wire once per run instead of once per call. A cancelled future means
        # a worker was restarted and lost it, so scatter again.
        if f is not self._pinned_fn or self._pinned_future.cancelled():
            self._pinned_future = self._client.scatter(f, broadcast=True)
            self._pinned_fn = f
        self._log_worker_memory()
        return self._client.gather(self._client.map(_apply, list(iterable), fn=self._pinned_future))

    def _log_worker_memory(self):
        """Log total and per-worker memory once per MCMC iteration."""
        # emcee maps twice per iteration, once for each half of the walkers.
        self._num_maps += 1
        if self._num_maps % 2 != 0:
            return
        workers = self._client.scheduler_info()['workers'].values()
        memory = [float(worker['metrics'].get('memory', 0)) for worker in workers]
        _log.info("Total memory (GB): %.3f, Min/max worker memory (GB): [%.3f, %.3f]", np.sum(memory)/1e9, np.amin(memory)/1e9, np.amax(memory)/1e9)

    def close(self):
        """Disconnect, and shut down the cluster if this pool started one."""
        self._client.close()
        if self._owned_cluster is not None:
            self._owned_cluster.close()


def make_scheduler(mcmc_settings, log_verbosity=0, log_filename=None):
    """
    Build the pool described by the ``mcmc`` settings of an ESPEI input file.

    Parameters
    ----------
    mcmc_settings : dict
        Validated ``mcmc`` settings. Only the ``scheduler`` and ``cores`` keys
        are read. ``scheduler`` is ``'multiprocessing'``, ``'dask'``, the path
        to a JSON scheduler file written by an externally managed dask
        scheduler, or None to run serially.
    log_verbosity : int
        Verbosity to configure ESPEI's logging with on dask workers.
    log_filename : str
        File for dask workers to log to.

    Returns
    -------
    An object with a ``map`` method, or None to let ``emcee`` use builtin ``map``.
    """
    scheduler = mcmc_settings['scheduler']
    cores = mcmc_settings.get('cores')
    if scheduler in ('multiprocessing', 'dask'):
        if cores is None:
            cores = multiprocessing.cpu_count()
        elif cores > multiprocessing.cpu_count():
            cores = multiprocessing.cpu_count()
            _log.warning("The number of cores chosen is larger than available. "
                         "Defaulting to run on the %s available cores.", cores)
        if scheduler == 'multiprocessing':
            return MultiprocessingPool(cores)
        return DaskPool(cores=cores, log_verbosity=log_verbosity, log_filename=log_filename)
    # Neither of the remaining schedulers sizes a pool of its own.
    if cores is not None:
        _log.warning("The 'cores' setting has no effect with the '%s' scheduler and is ignored.", scheduler)
    if scheduler is None:
        _log.info("Not using a parallel scheduler. ESPEI is running MCMC on a single core.")
        return None
    return DaskPool(scheduler_file=scheduler, log_verbosity=log_verbosity, log_filename=log_filename)

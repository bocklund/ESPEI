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
from concurrent.futures import ProcessPoolExecutor

_log = logging.getLogger(__name__)

__all__ = ["MultiprocessingPool"]


# Worker-process state. ``_PINNED_FN`` is installed once per worker by
# ``_install_fn`` so the (large) callable and its bound context cross the
# process boundary once per worker instead of once per ``map`` call.
# ``_INSTALL_COUNT`` exists so tests can assert that property from a worker.
_PINNED_FN = None
_INSTALL_COUNT = 0


def _install_fn(f):
    """
    Pin the mapped callable in this process. Runs once per worker at startup.

    Examples
    --------
    >>> _install_fn(abs)
    >>> _call_installed(-1.5)
    1.5

    """
    global _PINNED_FN, _INSTALL_COUNT
    _PINNED_FN = f
    _INSTALL_COUNT += 1


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

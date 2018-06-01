Installation
============


Anaconda (recommended)
----------------------

ESPEI does not require any special compiler, but several dependencies do.
Therefore it is suggested to install ESPEI from conda-forge.

.. code-block:: bash

    conda install -c pycalphad -c msys2 -c conda-forge --yes espei

After installation, you must turn off dask's work stealing.
Change the work stealing setting to ``work-stealing: False`` in ``~/.dask/config.yaml``.
See the `dask-distributed documentation <https://distributed.readthedocs.io/en/latest/configuration.html>`_ for more.

PyPI
----

Before you install ESPEI via PyPI, be aware that pycalphad and
emcee must be compiled and pycalphad requires an external
dependency of `Ipopt <https://projects.coin-or.org/Ipopt>`_.

.. code-block:: bash

    pip install espei

After installation, you must turn off dask's work stealing.
Change the work stealing setting to ``work-stealing: False`` in ``~/.dask/config.yaml``.
See the `dask-distributed documentation <https://distributed.readthedocs.io/en/latest/configuration.html>`_ for more.

Development versions
--------------------

You may install ESPEI however you like, but here we suggest using
Anaconda to download all of the required dependencies. This
method installs ESPEI with Anaconda, removes specifically the
ESPEI package, and replaces it with the package from GitHub.

.. code-block:: bash

    git clone https://github.com/phasesresearchlab/espei.git
    cd espei
    conda install espei
    conda remove --force espei
    pip install -e .

Upgrading ESPEI later requires you to run ``git pull`` in this directory.

After installation, you must turn off dask's work stealing.
Change the work stealing setting to ``work-stealing: False`` in ``~/.dask/config.yaml``.
See the `dask-distributed documentation <https://distributed.readthedocs.io/en/latest/configuration.html>`_ for more.
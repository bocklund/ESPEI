"""
Test espei.utils classes and functions.
"""
import pickle
import subprocess
import sys

import pytest
from tinydb import where
import espei.utils
from espei.utils import PickleableTinyDB, MemoryStorage, \
    bib_marker_map, extract_aliases

from .fixtures import datasets_db, tmp_file
from .testing_data import CU_MG_TDB


def test_immediate_client_is_deprecated():
    """espei.utils.ImmediateClient still resolves, but warns."""
    with pytest.warns(DeprecationWarning):
        cls = espei.utils.ImmediateClient
    assert cls is espei.parallel.DaskPool


# Run in a subprocess because other tests in this session may have imported
# distributed already.
_NO_DISTRIBUTED_SCRIPT = """
import sys
import espei.utils
assert "distributed" not in sys.modules, "importing espei.utils imported distributed"
espei.utils.ImmediateClient
assert "distributed" not in sys.modules, "espei.utils.ImmediateClient imported distributed"
"""


def test_importing_espei_utils_does_not_import_distributed():
    """espei.utils is the module that used to put distributed in every import chain."""
    proc = subprocess.run([sys.executable, "-c", _NO_DISTRIBUTED_SCRIPT],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr


def test_pickelable_tinydb_can_be_pickled_and_unpickled():
    """PickleableTinyDB should be able to be pickled and unpickled."""
    test_dict = {'test_key': ['test', 'values']}
    db = PickleableTinyDB(storage=MemoryStorage)
    db.insert(test_dict)
    db = pickle.loads(pickle.dumps(db))
    assert db.search(where('test_key').exists())[0] == test_dict


def test_bib_marker_map():
    """bib_marker_map should return a proper dict"""
    marker_dict = bib_marker_map(['otis2016', 'bocklund2018'])
    EXEMPLAR_DICT = {
        'bocklund2018': {
            'formatted': 'bocklund2018',
            'markers': {'fillstyle': 'none', 'marker': 'o'}
        },
        'otis2016': {
            'formatted': 'otis2016',
            'markers': {'fillstyle': 'none', 'marker': 'v'}
        }
    }
    assert EXEMPLAR_DICT == marker_dict


@pytest.mark.parametrize('reason, phase_models, expected_aliases', [
    (
        "No phases should give no aliases",
        {"phases": {}},
        {}
    ),
    (
        "A phase has an alias for itself and works without given aliases",
        {"phases": {"ALPHA": {}}},
        {"ALPHA": "ALPHA"}
    ),
    (
        "Empty aliases list works",
        {"phases": {"ALPHA": {"aliases": []}}},
        {"ALPHA": "ALPHA"}
    ),
    (
        "Basic test for adding aliases correctly",
        {"phases": {
            "ALPHA": {"aliases": ["FCC_A1"]}
        }},
        {"ALPHA": "ALPHA", "FCC_A1": "ALPHA"}
    ),
    (
        "A phase can have mulitple aliases",
        {"phases": {
            "ALPHA": {"aliases": ["FCC_A1", "A1", "FCC"]}
        }},
        {"ALPHA": "ALPHA", "FCC_A1": "ALPHA", "FCC": "ALPHA", "A1": "ALPHA"}
    ),
    (
        "Cannot have two phases with the same alias",
        {"phases": {
            "ALPHA": {"aliases": ["FCC_A1"]},
            "GAMMA": {"aliases": ["FCC_A1"]},
        }},
        None
    ),
    (
        "Cannot have a prescribed phase as an alias",
        {"phases": {
            "ALPHA": {"aliases": ["BETA"]},
            "BETA": {"aliases": []},
        }},
        None
    ),
]
)
def test_extract_aliases(reason, phase_models, expected_aliases):
    if expected_aliases is None:
        with pytest.raises(ValueError):
            aliases = extract_aliases(phase_models)
            print(aliases)
    else:
        assert extract_aliases(phase_models) == expected_aliases, reason

"""
Tests for fitting ordering energies of partitioned (order/disorder) models.

The tests use synthetic databases and datasets. The key property of the
ordering energy fit is that, for an exactly determined system (one datum
per symmetry-distinct ordered endmember configuration), the partitioned
model must reproduce the input data exactly.
"""

import numpy as np
import pytest
from tinydb import where
from pycalphad import Database, Model, variables as v

from espei.paramselect import generate_parameters
from espei.parameter_selection.fitting_descriptions import gibbs_energy_fitting_description
from espei.parameter_selection.fitting_steps import StepOrderingHM
from espei.parameter_selection.partitioned_ordering import (
    fit_ordering_parameters,
    generate_ordered_endmember_configurations,
    get_substitutional_sublattice_indices,
    infer_substitutional_symmetry,
)

from .fixtures import datasets_db

BCC_A2_TDB = """
ELEMENT MO BCC_A2 95.94 4589.0 28.56 !
ELEMENT NB BCC_A2 92.906 5220.0 36.27 !
FUNCTION GHSERMO 0.01 -10000-10*T; 6000 N !
FUNCTION GHSERNB 0.01 -12000-12*T; 6000 N !
TYPE_DEFINITION % SEQ * !
PHASE BCC_A2 % 1 1 !
CONSTITUENT BCC_A2 : MO,NB : !
PARAMETER G(BCC_A2,MO;0) 0.01 GHSERMO; 6000 N !
PARAMETER G(BCC_A2,NB;0) 0.01 GHSERNB; 6000 N !
PARAMETER L(BCC_A2,MO,NB;0) 0.01 -20000; 6000 N !
PARAMETER L(BCC_A2,MO,NB;1) 0.01 5000; 6000 N !
"""

BCC_A2_TERNARY_TDB = """
ELEMENT MO BCC_A2 95.94 4589.0 28.56 !
ELEMENT NB BCC_A2 92.906 5220.0 36.27 !
ELEMENT TA BCC_A2 180.95 5681.9 41.472 !
FUNCTION GHSERMO 0.01 -10000-10*T; 6000 N !
FUNCTION GHSERNB 0.01 -12000-12*T; 6000 N !
FUNCTION GHSERTA 0.01 -14000-14*T; 6000 N !
TYPE_DEFINITION % SEQ * !
PHASE BCC_A2 % 1 1 !
CONSTITUENT BCC_A2 : MO,NB,TA : !
PARAMETER G(BCC_A2,MO;0) 0.01 GHSERMO; 6000 N !
PARAMETER G(BCC_A2,NB;0) 0.01 GHSERNB; 6000 N !
PARAMETER G(BCC_A2,TA;0) 0.01 GHSERTA; 6000 N !
PARAMETER L(BCC_A2,MO,NB;0) 0.01 -20000; 6000 N !
PARAMETER L(BCC_A2,NB,TA;0) 0.01 8000; 6000 N !
"""

BCC_A2_INTERSTITIAL_TDB = """
ELEMENT VA VACUUM 0.0 0.0 0.0 !
ELEMENT C GRAPHITE 12.011 1054.0 5.7423 !
ELEMENT MO BCC_A2 95.94 4589.0 28.56 !
ELEMENT NB BCC_A2 92.906 5220.0 36.27 !
FUNCTION GHSERCC 0.01 -5000-5*T; 6000 N !
FUNCTION GHSERMO 0.01 -10000-10*T; 6000 N !
FUNCTION GHSERNB 0.01 -12000-12*T; 6000 N !
TYPE_DEFINITION % SEQ * !
PHASE BCC_A2 % 2 1 3 !
CONSTITUENT BCC_A2 : MO,NB : C,VA : !
PARAMETER G(BCC_A2,MO:VA;0) 0.01 GHSERMO; 6000 N !
PARAMETER G(BCC_A2,NB:VA;0) 0.01 GHSERNB; 6000 N !
PARAMETER G(BCC_A2,MO:C;0) 0.01 GHSERMO+3*GHSERCC+50000; 6000 N !
PARAMETER G(BCC_A2,NB:C;0) 0.01 GHSERNB+3*GHSERCC+60000; 6000 N !
PARAMETER L(BCC_A2,MO,NB:VA;0) 0.01 -20000; 6000 N !
"""

FCC_A1_TDB = """
ELEMENT AL FCC_A1 26.982 4577.3 28.322 !
ELEMENT NI FCC_A1 58.69 4787.0 29.796 !
FUNCTION GHSERAL 0.01 -8000-8*T; 6000 N !
FUNCTION GHSERNI 0.01 -9000-9*T; 6000 N !
TYPE_DEFINITION % SEQ * !
PHASE FCC_A1 % 1 1 !
CONSTITUENT FCC_A1 : AL,NI : !
PARAMETER G(FCC_A1,AL;0) 0.01 GHSERAL; 6000 N !
PARAMETER G(FCC_A1,NI;0) 0.01 GHSERNI; 6000 N !
PARAMETER L(FCC_A1,AL,NI;0) 0.01 -50000; 6000 N !
"""

# Canonical ordered occupancies for the binary 4 sublattice BCC model.
# Rows: D03 (A3B), B2 (A2B2), B32 (A2B2), D03 (AB3)
BCC_4SL_ORDERED_OCCUPANCIES = [
    [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
    [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
]


def add_ordered_phase(dbf, ordered_phase_name, disordered_phase_name, num_substitutional_sublattices, interstitial_site_ratios=None):
    """Add an ordered phase partitioned from the disordered phase, following
    the constituents of the disordered phase (first sublattice substitutional,
    the rest interstitial)."""
    interstitial_site_ratios = interstitial_site_ratios if interstitial_site_ratios is not None else []
    model_hints = {"ordered_phase": ordered_phase_name, "disordered_phase": disordered_phase_name}
    site_ratios = [1.0 / num_substitutional_sublattices] * num_substitutional_sublattices + list(interstitial_site_ratios)
    dbf.add_phase(ordered_phase_name, model_hints.copy(), site_ratios)
    dbf.phases[disordered_phase_name].model_hints.update(model_hints)
    dis_constituents = [sorted(sp.name for sp in subl) for subl in dbf.phases[disordered_phase_name].constituents]
    ord_constituents = [dis_constituents[0] for _ in range(num_substitutional_sublattices)] + dis_constituents[1:]
    dbf.add_phase_constituents(ordered_phase_name, ord_constituents)
    return dbf


def make_bcc_4sl_dbf():
    dbf = Database.from_string(BCC_A2_TDB, fmt="tdb")
    return add_ordered_phase(dbf, "BCC_4SL", "BCC_A2", 4)


def ordered_phase_enthalpy(dbf, comps, phase_name, occupancies, T=298.15, P=101325.0):
    """Return HM (J/mol-atom) of the phase at the site fractions given by
    occupancies, with the pure element reference (GHSER) set to zero, i.e. in
    the same reference state as HM_FORM data."""
    ghser_syms = {"GHSER" + (c.upper() * 2)[:2]: 0 for c in comps}
    mod = Model(dbf, comps, phase_name, parameters=ghser_syms)
    subs_dict = {v.T: T, v.P: P}
    constituents = [sorted(sp.name for sp in subl) for subl in mod.constituents]
    for subl_idx, subl_occupancies in enumerate(occupancies):
        if not isinstance(subl_occupancies, (list, tuple)):
            subl_occupancies = [subl_occupancies]
        for sp_name, occupancy in zip(constituents[subl_idx], subl_occupancies):
            subs_dict[v.Y(phase_name, subl_idx, sp_name)] = occupancy
    return float(mod.HM.subs(subs_dict))


def ordering_dataset(components, phase_name, occupancies, values, site_ratios=None, configurations=None):
    site_ratios = site_ratios if site_ratios is not None else [0.25, 0.25, 0.25, 0.25]
    if configurations is None:
        pure_els = [c for c in components if c != "VA"]
        configurations = [[pure_els for _ in range(len(occupancies[0]))] for _ in range(len(occupancies))]
    return {
        "components": list(components),
        "phases": [phase_name],
        "solver": {
            "mode": "manual",
            "sublattice_site_ratios": site_ratios,
            "sublattice_configurations": configurations,
            "sublattice_occupancies": occupancies,
        },
        "conditions": {"P": 101325, "T": 298.15},
        "output": "HM_FORM",
        "values": [[list(values)]],
    }


# ---------------------------------------------------------------------------
# Symmetry and configuration generation
# ---------------------------------------------------------------------------

def test_substitutional_sublattice_indices_bcc_4sl():
    dbf = make_bcc_4sl_dbf()
    assert get_substitutional_sublattice_indices(dbf, "BCC_4SL") == [0, 1, 2, 3]


def test_substitutional_sublattice_indices_with_interstitial():
    dbf = Database.from_string(BCC_A2_INTERSTITIAL_TDB, fmt="tdb")
    add_ordered_phase(dbf, "BCC_4SL", "BCC_A2", 4, interstitial_site_ratios=[3])
    assert get_substitutional_sublattice_indices(dbf, "BCC_4SL") == [0, 1, 2, 3]


def test_infer_symmetry_bcc_4sl():
    dbf = make_bcc_4sl_dbf()
    assert infer_substitutional_symmetry(dbf, "BCC_4SL") == [[0, 1], [2, 3]]


def test_infer_symmetry_fcc_4sl():
    dbf = Database.from_string(FCC_A1_TDB, fmt="tdb")
    add_ordered_phase(dbf, "FCC_L12", "FCC_A1", 4)
    assert infer_substitutional_symmetry(dbf, "FCC_L12") == [[0, 1, 2, 3]]


def test_infer_symmetry_2sl_equal_site_ratios():
    dbf = Database.from_string(BCC_A2_TDB, fmt="tdb")
    dbf.add_phase("BCC_B2", {"ordered_phase": "BCC_B2", "disordered_phase": "BCC_A2"}, [0.5, 0.5])
    dbf.phases["BCC_A2"].model_hints.update({"ordered_phase": "BCC_B2", "disordered_phase": "BCC_A2"})
    dbf.add_phase_constituents("BCC_B2", [["MO", "NB"], ["MO", "NB"]])
    assert infer_substitutional_symmetry(dbf, "BCC_B2") == [[0, 1]]


def test_infer_symmetry_2sl_unequal_site_ratios():
    dbf = Database.from_string(FCC_A1_TDB, fmt="tdb")
    dbf.add_phase("FCC_L12", {"ordered_phase": "FCC_L12", "disordered_phase": "FCC_A1"}, [0.75, 0.25])
    dbf.phases["FCC_A1"].model_hints.update({"ordered_phase": "FCC_L12", "disordered_phase": "FCC_A1"})
    dbf.add_phase_constituents("FCC_L12", [["AL", "NI"], ["AL", "NI"]])
    assert infer_substitutional_symmetry(dbf, "FCC_L12") is None


def test_infer_symmetry_unknown_parent_raises():
    dbf = Database.from_string(BCC_A2_TDB, fmt="tdb")
    # rename to something that doesn't contain a known parent phase prefix
    dbf.add_phase("MYSTERY_4SL", {"ordered_phase": "MYSTERY_4SL", "disordered_phase": "BCC_A2"}, [0.25] * 4)
    dbf.add_phase_constituents("MYSTERY_4SL", [["MO", "NB"]] * 4)
    # give the disordered phase an unrecognizable name by making a fresh database
    dbf.phases["MYSTERY_4SL"].model_hints["disordered_phase"] = "MYSTERY_DIS"
    dbf.add_phase("MYSTERY_DIS", {}, [1])
    dbf.add_phase_constituents("MYSTERY_DIS", [["MO", "NB"]])
    with pytest.raises(ValueError):
        infer_substitutional_symmetry(dbf, "MYSTERY_4SL")


def test_ordered_endmember_configurations_bcc_4sl():
    dbf = make_bcc_4sl_dbf()
    configs = generate_ordered_endmember_configurations(dbf, "BCC_4SL", ["MO", "NB"], [[0, 1], [2, 3]])
    expected = [
        ("MO", "MO", "MO", "NB"),  # D03: A3B
        ("MO", "MO", "NB", "NB"),  # B2
        ("MO", "NB", "MO", "NB"),  # B32
        ("MO", "NB", "NB", "NB"),  # D03: AB3
    ]
    assert sorted(configs) == sorted(expected)


def test_ordered_endmember_configurations_fcc_4sl():
    dbf = Database.from_string(FCC_A1_TDB, fmt="tdb")
    add_ordered_phase(dbf, "FCC_L12", "FCC_A1", 4)
    configs = generate_ordered_endmember_configurations(dbf, "FCC_L12", ["AL", "NI"], [[0, 1, 2, 3]])
    expected = [
        ("AL", "AL", "AL", "NI"),  # L12: A3B
        ("AL", "AL", "NI", "NI"),  # L10
        ("AL", "NI", "NI", "NI"),  # L12: AB3
    ]
    assert sorted(configs) == sorted(expected)


def test_ordered_endmember_configurations_carry_interstitial_sublattice():
    dbf = Database.from_string(BCC_A2_INTERSTITIAL_TDB, fmt="tdb")
    add_ordered_phase(dbf, "BCC_4SL", "BCC_A2", 4, interstitial_site_ratios=[3])
    configs = generate_ordered_endmember_configurations(dbf, "BCC_4SL", ["MO", "NB"], [[0, 1], [2, 3]])
    # interstitial constituents are restricted to the components being fit
    # plus VA, so no configurations with C should be generated
    expected = [
        ("MO", "MO", "MO", "NB", "VA"),
        ("MO", "MO", "NB", "NB", "VA"),
        ("MO", "NB", "MO", "NB", "VA"),
        ("MO", "NB", "NB", "NB", "VA"),
    ]
    assert sorted(configs) == sorted(expected)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def test_fit_ordering_parameters_bcc_4sl_exactly_reproduces_data(datasets_db):
    dbf = make_bcc_4sl_dbf()
    input_values = [-6000.0, -9000.0, -5500.0, -4000.0]
    datasets_db.insert(ordering_dataset(["MO", "NB"], "BCC_4SL", BCC_4SL_ORDERED_OCCUPANCIES, input_values))

    fit_ordering_parameters(dbf, "BCC_4SL", datasets_db)

    # 4 symmetry-distinct parameters, expanded to 4 + 2 + 4 + 4 = 14 database parameters (D03 + B2 + B32 + D03)
    params = dbf._parameters.search((where("phase_name") == "BCC_4SL") & (where("parameter_type") == "G"))
    assert len(params) == 14
    vv_symbols = sorted(name for name in dbf.symbols if name.startswith("VV"))
    assert len(vv_symbols) == 4
    # all parameter values are concrete floats
    for name in vv_symbols:
        assert isinstance(dbf.symbols[name], float)
    # the fit is exactly determined, so the model must reproduce the data
    for occupancies, expected in zip(BCC_4SL_ORDERED_OCCUPANCIES, input_values):
        assert np.isclose(ordered_phase_enthalpy(dbf, ["MO", "NB"], "BCC_4SL", occupancies), expected, atol=1.0)
    # disordered phase parameters are untouched
    assert len(dbf._parameters.search(where("phase_name") == "BCC_A2")) == 4


def test_fit_ordering_parameters_2sl_b2_exactly_reproduces_data(datasets_db):
    dbf = Database.from_string(BCC_A2_TDB, fmt="tdb")
    hints = {"ordered_phase": "BCC_B2", "disordered_phase": "BCC_A2"}
    dbf.add_phase("BCC_B2", hints.copy(), [0.5, 0.5])
    dbf.phases["BCC_A2"].model_hints.update(hints)
    dbf.add_phase_constituents("BCC_B2", [["MO", "NB"], ["MO", "NB"]])

    occupancies = [[[1.0, 0.0], [0.0, 1.0]]]
    input_values = [-7000.0]
    datasets_db.insert(ordering_dataset(["MO", "NB"], "BCC_B2", occupancies, input_values, site_ratios=[0.5, 0.5]))

    fit_ordering_parameters(dbf, "BCC_B2", datasets_db)

    # 1 symmetry-distinct parameter, expanded to 2 database parameters (MO:NB and NB:MO)
    params = dbf._parameters.search((where("phase_name") == "BCC_B2") & (where("parameter_type") == "G"))
    assert len(params) == 2
    vv_symbols = sorted(name for name in dbf.symbols if name.startswith("VV"))
    assert len(vv_symbols) == 1
    assert np.isclose(ordered_phase_enthalpy(dbf, ["MO", "NB"], "BCC_B2", occupancies[0]), input_values[0], atol=1.0)


def test_fit_ordering_parameters_with_interstitial_sublattice(datasets_db):
    dbf = Database.from_string(BCC_A2_INTERSTITIAL_TDB, fmt="tdb")
    add_ordered_phase(dbf, "BCC_4SL", "BCC_A2", 4, interstitial_site_ratios=[3])

    occupancies = [subl_occ + [1.0] for subl_occ in BCC_4SL_ORDERED_OCCUPANCIES]
    configurations = [[["MO", "NB"]] * 4 + ["VA"] for _ in range(4)]
    input_values = [-4000.0, -7000.0, -4500.0, -3500.0]
    datasets_db.insert(ordering_dataset(
        ["MO", "NB", "VA"], "BCC_4SL", occupancies, input_values,
        site_ratios=[0.25, 0.25, 0.25, 0.25, 3], configurations=configurations,
    ))

    fit_ordering_parameters(dbf, "BCC_4SL", datasets_db)

    params = dbf._parameters.search((where("phase_name") == "BCC_4SL") & (where("parameter_type") == "G"))
    assert len(params) == 14
    vv_symbols = sorted(name for name in dbf.symbols if name.startswith("VV"))
    assert len(vv_symbols) == 4
    for occ, expected in zip(occupancies, input_values):
        assert np.isclose(ordered_phase_enthalpy(dbf, ["MO", "NB", "VA"], "BCC_4SL", occ), expected, atol=1.0)


def test_fit_ordering_parameters_multiple_binary_subsystems(datasets_db):
    dbf = Database.from_string(BCC_A2_TERNARY_TDB, fmt="tdb")
    add_ordered_phase(dbf, "BCC_4SL", "BCC_A2", 4)

    mo_nb_values = [-6000.0, -9000.0, -5500.0, -4000.0]
    nb_ta_values = [1000.0, 2000.0, 1500.0, 800.0]
    datasets_db.insert(ordering_dataset(["MO", "NB"], "BCC_4SL", BCC_4SL_ORDERED_OCCUPANCIES, mo_nb_values))
    datasets_db.insert(ordering_dataset(["NB", "TA"], "BCC_4SL", BCC_4SL_ORDERED_OCCUPANCIES, nb_ta_values))

    fit_ordering_parameters(dbf, "BCC_4SL", datasets_db)

    # two binary subsystems fit, no data for MO-TA so no parameters for it
    params = dbf._parameters.search((where("phase_name") == "BCC_4SL") & (where("parameter_type") == "G"))
    assert len(params) == 28
    vv_symbols = sorted(name for name in dbf.symbols if name.startswith("VV"))
    assert len(vv_symbols) == 8
    for occ, expected in zip(BCC_4SL_ORDERED_OCCUPANCIES, mo_nb_values):
        assert np.isclose(ordered_phase_enthalpy(dbf, ["MO", "NB"], "BCC_4SL", occ), expected, atol=1.0)
    for occ, expected in zip(BCC_4SL_ORDERED_OCCUPANCIES, nb_ta_values):
        assert np.isclose(ordered_phase_enthalpy(dbf, ["NB", "TA"], "BCC_4SL", occ), expected, atol=1.0)


def test_fit_ordering_parameters_no_data_is_a_no_op(datasets_db):
    dbf = make_bcc_4sl_dbf()
    fit_ordering_parameters(dbf, "BCC_4SL", datasets_db)
    assert len(dbf._parameters.search(where("phase_name") == "BCC_4SL")) == 0
    assert len([name for name in dbf.symbols if name.startswith("VV")]) == 0


def test_fit_ordering_parameters_requires_partitioned_phase(datasets_db):
    dbf = Database.from_string(BCC_A2_TDB, fmt="tdb")
    with pytest.raises(ValueError):
        fit_ordering_parameters(dbf, "BCC_A2", datasets_db)


# ---------------------------------------------------------------------------
# Integration with parameter generation
# ---------------------------------------------------------------------------

def test_ordering_step_included_in_default_fitting_description():
    assert StepOrderingHM in gibbs_energy_fitting_description.fitting_steps


def test_generate_parameters_fits_partitioned_ordered_phase(datasets_db):
    # The BCC_4SL phase sorts before BCC_A2 alphabetically, but the disordered
    # phase must be fit first for the ordering energies to be correct.
    dbf_in = Database()
    dbf_in.elements.update({"V", "W"})
    dbf_in.species.update({v.Species("V"), v.Species("W")})
    hints = {"ordered_phase": "BCC_4SL", "disordered_phase": "BCC_A2"}
    dbf_in.add_phase("BCC_A2", hints.copy(), [1])
    dbf_in.add_phase_constituents("BCC_A2", [["V", "W"]])
    dbf_in.add_phase("BCC_4SL", hints.copy(), [0.25] * 4)
    dbf_in.add_phase_constituents("BCC_4SL", [["V", "W"]] * 4)

    phase_models = {
        "components": ["V", "W"],
        "phases": {
            "BCC_A2": {"sublattice_model": [["V", "W"]], "sublattice_site_ratios": [1]},
            "BCC_4SL": {"sublattice_model": [["V", "W"]] * 4, "sublattice_site_ratios": [0.25] * 4},
        },
    }
    dataset_disordered_mixing = {
        "components": ["V", "W"],
        "phases": ["BCC_A2"],
        "solver": {
            "mode": "manual",
            "sublattice_site_ratios": [1],
            "sublattice_configurations": [[["V", "W"]]],
            "sublattice_occupancies": [[[0.5, 0.5]]],
        },
        "conditions": {"P": 101325, "T": 298.15},
        "output": "HM_MIX",
        "values": [[[-12000.0]]],
    }
    input_values = [-6000.0, -9000.0, -5500.0, -4000.0]
    datasets_db.insert(dataset_disordered_mixing)
    datasets_db.insert(ordering_dataset(["V", "W"], "BCC_4SL", BCC_4SL_ORDERED_OCCUPANCIES, input_values))

    dbf = generate_parameters(phase_models, datasets_db, "SGTE91", "linear", dbf=dbf_in)

    # the disordered mixing parameter is fit (and first, so it gets VV0000)
    assert dbf.symbols["VV0000"] == -48000.0
    assert len(dbf._parameters.search((where("phase_name") == "BCC_A2") & (where("parameter_type") == "L"))) == 1
    # the ordering parameters are fit
    params = dbf._parameters.search((where("phase_name") == "BCC_4SL") & (where("parameter_type") == "G"))
    assert len(params) == 14
    vv_symbols = sorted(name for name in dbf.symbols if name.startswith("VV"))
    assert vv_symbols == ["VV0000", "VV0001", "VV0002", "VV0003", "VV0004"]
    # the partitioned model reproduces the ordering data
    for occ, expected in zip(BCC_4SL_ORDERED_OCCUPANCIES, input_values):
        assert np.isclose(ordered_phase_enthalpy(dbf, ["V", "W"], "BCC_4SL", occ), expected, atol=1.0)
    # the database round-trips through TDB, preserving the partitioned model
    read_dbf = Database.from_string(dbf.to_string(fmt="tdb"), fmt="tdb")
    assert read_dbf.phases["BCC_4SL"].model_hints["disordered_phase"] == "BCC_A2"
    assert read_dbf.phases["BCC_4SL"].model_hints["ordered_phase"] == "BCC_4SL"
    for occ, expected in zip(BCC_4SL_ORDERED_OCCUPANCIES, input_values):
        assert np.isclose(ordered_phase_enthalpy(read_dbf, ["V", "W"], "BCC_4SL", occ), expected, atol=1.0)


def test_generate_parameters_normal_phases_unchanged_by_ordering_step(datasets_db):
    """A non-partitioned phase is fit as before with the ordering step in the default description."""
    phase_models = {
        "components": ["AL", "B"],
        "phases": {
            "FCC_A1": {"sublattice_model": [["AL", "B"]], "sublattice_site_ratios": [1]},
        },
    }
    dataset_excess_mixing = {
        "components": ["AL", "B"],
        "phases": ["FCC_A1"],
        "solver": {
            "mode": "manual",
            "sublattice_site_ratios": [1],
            "sublattice_configurations": [[["AL", "B"]]],
            "sublattice_occupancies": [[[0.5, 0.5]]],
        },
        "conditions": {"P": 101325, "T": 298.15},
        "output": "HM_MIX",
        "values": [[[-10000.0]]],
    }
    datasets_db.insert(dataset_excess_mixing)
    dbf = generate_parameters(phase_models, datasets_db, "SGTE91", "linear")
    assert len(dbf._parameters.search(where("parameter_type") == "L")) == 1
    assert dbf.symbols["VV0000"] == -40000
    assert len(dbf._parameters.search((where("phase_name") == "FCC_A1") & (where("parameter_type") == "G"))) == 2

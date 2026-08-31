"""
Fit the ordering energies of partitioned (order/disorder) two-part models.

In a partitioned model, the Gibbs energy of the ordered phase is

.. math::

    G = G^\\mathrm{dis}(x) + \\Delta G^\\mathrm{ord}(y)

where :math:`\\Delta G^\\mathrm{ord}(y) = G^\\mathrm{ord}(y) - G^\\mathrm{ord}(y = x)`
following Connetable *et al.* (see ``pycalphad.Model.atomic_ordering_energy``).
The endmember Gibbs energy parameters of the ordered phase are the ordering
energies. Because :math:`G^\\mathrm{ord}(y = x)` couples every ordering
parameter at every ordered configuration, all symmetry-distinct ordered
endmember parameters of a subsystem must be fit simultaneously by solving a
linear system relating the parameters to the difference between the energies
of the ordered configurations and the disordered phase, following the approach
of Davey, Malinov and Coakley, Calphad 71 (2020) 101999 (in particular the
supplemental material).

The strategy is:

1. Start with a Database where the disordered phase is already fit and the
   ordered phase is defined (constituents and ``ordered_phase``/``disordered_phase``
   model hints) but has no parameters.
2. Determine the symmetry relationship between the substitutional sublattices
   of the ordered phase (user-supplied or inferred for known parent phases).
3. For each binary subsystem with data, generate the symmetry-distinct ordered
   endmember configurations and add symbolic parameters for them.
4. Build the linear system relating the parameters (via the atomic ordering
   energy contribution) to the enthalpy data of the ordered configurations
   with the fitted disordered energy subtracted, and solve it.

Only substitutional partitioned models are fit. Interstitial sublattices are
carried forward (with their constituents restricted to the components being
fit plus VA), but ordering of interstitial constituents is not fit. Interaction
(excess) parameters of the ordered phase are not fit.
"""

import itertools
import logging

import numpy as np
from symengine import Symbol
from tinydb import where

from espei.core_utils import get_prop_data, filter_temperatures
from espei.error_functions.non_equilibrium_thermochemical_error import get_prop_samples, get_sample_condition_dicts
from espei.parameter_selection.fitting_descriptions import ModelFittingDescription, gibbs_energy_fitting_description
from espei.sublattice_tools import canonical_sort_key, generate_endmembers, generate_symmetric_group, tuplify
from espei.utils import sigfigs

_log = logging.getLogger(__name__)

# Hardcoded substitutional sublattice symmetries for 4 sublattice ordered
# phases of known parent phases, keyed by a prefix of the phase name.
# * BCC: the two pairs of sublattices (first and second nearest neighbor
#   tetrahedron sites) are internally equivalent, describing B2, B32, and D0_3
#   ordering.
# * FCC and HCP: all four sublattices (nearest neighbor tetrahedron sites) are
#   equivalent, describing L1_0 and L1_2 (FCC) or B19 and D0_19 (HCP) ordering.
FOUR_SUBLATTICE_SUBSTITUTIONAL_SYMMETRIES = {
    "BCC": [[0, 1], [2, 3]],
    "FCC": [[0, 1, 2, 3]],
    "HCP": [[0, 1, 2, 3]],
}


def _validate_ordered_phase(dbf, phase_name):
    """Raise a ValueError if the phase is not the ordered phase of a supported partitioned model."""
    phase_obj = dbf.phases[phase_name]
    if phase_obj.model_hints.get("ordered_phase") != phase_name:
        raise ValueError(f"{phase_name} is not the ordered phase of a partitioned model. The `ordered_phase` and `disordered_phase` model hints must be set.")
    disordered_phase_name = phase_obj.model_hints.get("disordered_phase")
    if disordered_phase_name not in dbf.phases:
        raise ValueError(f"The disordered phase {disordered_phase_name} of the partitioned phase {phase_name} is not in the Database.")
    if phase_obj.model_hints.get("never_disorder", False):
        raise NotImplementedError(f"Fitting ordering energies of phases with the `never_disorder` model hint ({phase_name}) is not supported.")


def get_substitutional_sublattice_indices(dbf, phase_name):
    """
    Return the indices of the substitutional (ordering) sublattices of an
    ordered phase in a partitioned model.

    Following pycalphad, the first sublattice of the disordered phase is
    assumed to be the substitutional sublattice and the sublattices of the
    ordered phase with exactly the same constituents are the ordering
    sublattices. All other sublattices are interstitial.

    Parameters
    ----------
    dbf : Database
        pycalphad Database.
    phase_name : str
        Name of the ordered phase of a partitioned model.

    Returns
    -------
    [int]

    """
    _validate_ordered_phase(dbf, phase_name)
    phase_obj = dbf.phases[phase_name]
    disordered_phase_name = phase_obj.model_hints["disordered_phase"]
    disordered_substitutional_constituents = dbf.phases[disordered_phase_name].constituents[0]
    return [
        idx for idx, constituents in enumerate(phase_obj.constituents)
        if len(set(constituents).symmetric_difference(disordered_substitutional_constituents)) == 0
    ]


def infer_substitutional_symmetry(dbf, phase_name):
    """
    Return the symmetry (list of lists of equivalent sublattice indices) of
    the substitutional sublattices of an ordered phase in a partitioned model.

    Two sublattice models are symmetric if the substitutional site ratios are
    equal (e.g. B2) and have no symmetry otherwise (e.g. L1_2 described by 2
    sublattices). Four sublattice models use hardcoded symmetries for known
    parent phases (BCC, FCC, HCP), matched by prefixes of the disordered and
    ordered phase names.

    Parameters
    ----------
    dbf : Database
        pycalphad Database.
    phase_name : str
        Name of the ordered phase of a partitioned model.

    Returns
    -------
    Union[None, [[int]]]

    """
    phase_obj = dbf.phases[phase_name]
    substitutional_idxs = get_substitutional_sublattice_indices(dbf, phase_name)
    if len(substitutional_idxs) == 2:
        site_ratios = [phase_obj.sublattices[idx] for idx in substitutional_idxs]
        if np.isclose(site_ratios[0], site_ratios[1]):
            return [substitutional_idxs]
        else:
            return None
    elif len(substitutional_idxs) == 4:
        disordered_phase_name = phase_obj.model_hints["disordered_phase"]
        for name in (disordered_phase_name, phase_name):
            for parent_prefix, parent_symmetry in FOUR_SUBLATTICE_SUBSTITUTIONAL_SYMMETRIES.items():
                if name.upper().startswith(parent_prefix):
                    return [[substitutional_idxs[i] for i in group] for group in parent_symmetry]
        raise ValueError(
            f"Unable to infer the substitutional sublattice symmetry for the ordered phase {phase_name} "
            f"(disordered phase {disordered_phase_name}) because neither phase name starts with a known "
            f"parent phase prefix ({sorted(FOUR_SUBLATTICE_SUBSTITUTIONAL_SYMMETRIES.keys())}). "
            "Specify the symmetry explicitly (`equivalent_sublattices` in the phase models)."
        )
    else:
        raise ValueError(
            f"Only 2 and 4 substitutional sublattice partitioned models are supported for inferring "
            f"sublattice symmetry, got {len(substitutional_idxs)} substitutional sublattices for {phase_name}. "
            "Specify the symmetry explicitly (`equivalent_sublattices` in the phase models)."
        )


def generate_ordered_endmember_configurations(dbf, phase_name, components, symmetry):
    """
    Return the symmetry-distinct ordered endmember configurations of an
    ordered phase for the given components.

    Configurations where all substitutional sublattices have the same
    constituent (i.e. disordered endmembers) are excluded, since their
    ordering energies are zero by construction of the partitioned model.
    Interstitial sublattice constituents are restricted to the given
    components plus VA.

    Parameters
    ----------
    dbf : Database
        pycalphad Database.
    phase_name : str
        Name of the ordered phase of a partitioned model.
    components : [str]
        Names of the (pure element) components to consider, e.g. a binary pair.
    symmetry : Union[None, [[int]]]
        Symmetry of the substitutional sublattices.

    Returns
    -------
    [tuple]
        Sorted list of one configuration (a tuple of constituent names) per
        symmetry-distinct ordered endmember.

    """
    phase_obj = dbf.phases[phase_name]
    substitutional_idxs = set(get_substitutional_sublattice_indices(dbf, phase_name))
    substitutional_active = sorted(set(components) - {"VA"})
    interstitial_active = sorted(set(components) | {"VA"})
    sublattice_model = []
    for idx, constituents in enumerate(phase_obj.constituents):
        constituent_names = {sp.name for sp in constituents}
        if idx in substitutional_idxs:
            active = sorted(constituent_names.intersection(substitutional_active))
        else:
            active = sorted(constituent_names.intersection(interstitial_active))
        if len(active) == 0:
            # cannot construct any configuration for these components
            return []
        sublattice_model.append(active)
    candidate_configs = []
    seen_configs = set()
    for endmember in generate_endmembers(sublattice_model, symmetry):
        if endmember in seen_configs:
            continue
        # `generate_endmembers` canonicalizes within each symmetric group of
        # sublattices, but does not deduplicate configurations that are
        # equivalent by interchanging the groups, so we deduplicate with the
        # full symmetric group here.
        symmetric_group = generate_symmetric_group(endmember, symmetry)
        seen_configs.update(symmetric_group)
        if len({endmember[idx] for idx in substitutional_idxs}) > 1:
            candidate_configs.append(symmetric_group[0])
    return sorted(candidate_configs, key=canonical_sort_key)


def _fit_ordering_parameters_subsystem(dbf, phase_name, components, symmetry, datasets, ordering_steps, model_class):
    """Fit the ordering energies of one subsystem (usually a binary pair) of an ordered phase, modifying the Database in place."""
    # Lazy imports to avoid a circular import of espei.paramselect
    from espei.paramselect import get_next_symbol, _build_feature_matrix, _param_present_in_database

    phase_obj = dbf.phases[phase_name]
    active_species = {sp for subl in phase_obj.constituents for sp in subl}
    comps = sorted(set(components) | ({"VA"} if "VA" in {sp.name for sp in active_species} else set()))
    candidate_configs = generate_ordered_endmember_configurations(dbf, phase_name, components, symmetry)
    if len(candidate_configs) == 0:
        _log.trace('No ordered endmember configurations can be generated for %s: %s', phase_name, components)
        return
    config_tup = tuple(tuple(sorted({sp.name for sp in constituents}.intersection(comps))) for constituents in phase_obj.constituents)
    for fitting_step in ordering_steps:
        _log.trace('Fitting step: %s', fitting_step)
        new_configs = [config for config in candidate_configs if not _param_present_in_database(dbf, phase_name, config, fitting_step.parameter_name)]
        if len(new_configs) == 0:
            _log.trace('All ordering parameters already in the database for %s: %s. Skipping.', phase_name, components)
            continue
        # Search for relevant data
        desired_props = [fitting_step.data_types_read + refstate for refstate in fitting_step.supported_reference_states]
        desired_data = get_prop_data(comps, phase_name, desired_props, datasets, additional_query=where('solver').exists())
        desired_data = filter_temperatures(desired_data)
        _log.trace('%s: datasets found: %s', desired_props, len(desired_data))
        if len(desired_data) == 0:
            continue
        # The response vector is the data with the disordered part of the
        # partitioned model subtracted, i.e. the target ordering energies.
        # The model must be built before the ordering parameters are added so
        # that the atomic ordering contribution contains only parameters that
        # are already fit.
        fixed_model = model_class(dbf, comps, phase_name, parameters={'GHSER'+(c.upper()*2)[:2]: 0 for c in comps})
        calculate_dict = get_prop_samples(desired_data, config_tup)
        sample_condition_dicts = get_sample_condition_dicts(calculate_dict, config_tup, phase_name)
        response_vector = fitting_step.get_response_vector(fixed_model, [0], desired_data, sample_condition_dicts)
        # Add symbolic parameters for the new configurations and build the
        # feature matrix from the atomic ordering energy contribution, which
        # is linear in the parameters.
        symbol_names = []
        for config in new_configs:
            symbol_name = get_next_symbol(dbf)
            dbf.symbols[symbol_name] = 0.0  # placeholder value, overwritten below
            symbol_names.append(symbol_name)
            for symmetric_config in generate_symmetric_group(config, symmetry):
                dbf.add_parameter(fitting_step.parameter_name, phase_name, tuple(map(tuplify, symmetric_config)), 0, Symbol(symbol_name))
        # Passing a list of parameters keeps the symbols symbolic in the model
        ordering_model = model_class(dbf, comps, phase_name, parameters=symbol_names)
        ordering_energy = ordering_model.models['ord']
        features = [fitting_step.transform_feature(ordering_energy.diff(Symbol(symbol_name))) for symbol_name in symbol_names]
        feature_matrix = _build_feature_matrix(sample_condition_dicts, features)
        weights = np.asarray(calculate_dict['weights'], dtype=np.float64)
        matrix_rank = np.linalg.matrix_rank(feature_matrix)
        if matrix_rank < len(symbol_names):
            _log.warning(
                'The ordering energy system for phase %s of subsystem %s is rank deficient '
                '(rank %s < %s parameters). The data do not constrain all ordering parameters '
                'and a minimum norm solution will be used.', phase_name, components, matrix_rank, len(symbol_names),
            )
        parameter_values, *_ = np.linalg.lstsq(feature_matrix * weights[:, None], response_vector * weights, rcond=None)
        _log.trace('Fit ordering parameters for %s: %s', phase_name, dict(zip(new_configs, parameter_values)))
        for symbol_name, value in zip(symbol_names, parameter_values):
            dbf.symbols[symbol_name] = sigfigs(value, 6)


def fit_ordering_parameters(dbf, phase_name, datasets, symmetry=None, fitting_description: ModelFittingDescription = gibbs_energy_fitting_description):
    """
    Fit the ordering energies (ordered endmember parameters) of the ordered
    phase of a partitioned (order/disorder) model.

    The disordered phase must already be fit. All binary subsystems of the
    substitutional constituents with matching data are fit, one subsystem at
    a time.

    Parameters
    ----------
    dbf : Database
        pycalphad Database with the disordered phase fit and the ordered phase
        defined (constituents and ``ordered_phase``/``disordered_phase`` model
        hints), to be modified in place.
    phase_name : str
        Name of the ordered phase of the partitioned model.
    datasets : PickleableTinyDB
        All the datasets desired to fit to.
    symmetry : Union[None, [[int]]]
        Symmetry of the sublattice configuration. If None (the default), the
        symmetry of the substitutional sublattices will be inferred, see
        ``infer_substitutional_symmetry``.
    fitting_description : ModelFittingDescription
        ModelFittingDescription object describing the fitting steps and model.
        Only ordering fitting steps (``is_ordering_step == True``) are used.

    Returns
    -------
    None
        Modifies the dbf.

    """
    _validate_ordered_phase(dbf, phase_name)
    ordering_steps = [step for step in fitting_description.fitting_steps if step.is_ordering_step]
    if len(ordering_steps) == 0:
        _log.warning('No ordering fitting steps in the fitting description. No ordering parameters will be fit for %s.', phase_name)
        return
    if symmetry is None:
        symmetry = infer_substitutional_symmetry(dbf, phase_name)
    if not hasattr(dbf, 'varcounter'):
        dbf.varcounter = 0
    disordered_phase_name = dbf.phases[phase_name].model_hints["disordered_phase"]
    substitutional_constituents = dbf.phases[disordered_phase_name].constituents[0]
    pure_elements = sorted({sp.name for sp in substitutional_constituents if sp.number_of_atoms > 0}.intersection(dbf.elements))
    for subsystem_components in itertools.combinations(pure_elements, 2):
        _log.trace('Fitting ordering energies for %s: %s', phase_name, subsystem_components)
        _fit_ordering_parameters_subsystem(dbf, phase_name, list(subsystem_components), symmetry, datasets, ordering_steps, fitting_description.model)

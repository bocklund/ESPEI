"""
Calculate error due to thermochemical quantities: heat capacity, entropy, enthalpy.
"""

import logging
from collections import OrderedDict

import sympy
from scipy.stats import norm
import numpy as np
from tinydb import where

from pycalphad import Model, ReferenceState, variables as v
from pycalphad.core.utils import unpack_components, get_pure_elements

from espei.core_utils import ravel_conditions, get_prop_data
from espei.utils import database_symbols_to_fit
from espei.error_functions.zpf_error import update_phase_record_parameters
from espei.shadow_functions import calculate_
from pycalphad.codegen.callables import build_phase_records


def calculate_points_array(phase_constituents, configuration, occupancies=None):
    """
    Calculate the points array to use in pycalphad calculate calls.

    Converts the configuration data (and occupancies for mixing data) into the
    points array by looking up the indices in the active phase constituents.

    Parameters
    ----------
    phase_constituents : list
        List of active constituents in a phase
    configuration : list
        List of the sublattice configuration
    occupancies : list
        List of sublattice occupancies. Required for mixing sublattices, otherwise takes no effect.

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    Errors will be raised if components in the configuration are not in the
    corresponding phase constituents sublattice.
    """
    # pad the occupancies for zipping if none were passed (the case for non-mixing)
    if occupancies is None:
        occupancies = [0] * len(configuration)

    # construct the points array from zeros
    points = np.zeros(sum(len(subl) for subl in phase_constituents))
    current_subl_idx = 0  # index that marks the beginning of the sublattice
    for phase_subl, config_subl, subl_occupancies in zip(phase_constituents, configuration, occupancies):
        phase_subl = list(phase_subl)
        if isinstance(config_subl, (tuple, list)):
            # we have mixing on the sublattice
            for comp, comp_occupancy in zip(config_subl, subl_occupancies):
                points[current_subl_idx + phase_subl.index(comp)] = comp_occupancy
        else:
            points[current_subl_idx + phase_subl.index(config_subl)] = 1
        current_subl_idx += len(phase_subl)
    return points


def get_prop_samples(dbf, comps, phase_name, desired_data):
    """
    Return data values and the conditions to calculate them by pycalphad
    calculate from the datasets

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    phase_name : str
        Name of the phase to consider from the Database
    desired_data : list
        List of dictionary datasets that contain the values to sample

    Returns
    -------
    dict
        Dictionary of condition kwargs for pycalphad's calculate and the expected values

    """
    # TODO: assumes T, P as conditions
    # sublattice constituents are Species objects, so we need to be doing intersections with those
    species_comps = unpack_components(dbf, comps)
    phase_constituents = dbf.phases[phase_name].constituents
    # phase constituents must be filtered to only active:
    phase_constituents = [[c.name for c in sorted(subl_constituents.intersection(set(species_comps)))] for subl_constituents in phase_constituents]

    # calculate needs points, state variable lists, and values to compare to
    calculate_dict = {
        'P': np.array([]),
        'T': np.array([]),
        'points': np.atleast_2d([[]]).reshape(-1, sum([len(subl) for subl in phase_constituents])),
        'values': np.array([]),
        'weights': [],
        'references': [],
    }

    for datum in desired_data:
        # extract the data we care about
        datum_T = datum['conditions']['T']
        datum_P = datum['conditions']['P']
        configurations = datum['solver']['sublattice_configurations']
        occupancies = datum['solver'].get('sublattice_occupancies')
        values = np.array(datum['values'])

        # broadcast and flatten the conditions arrays
        P, T = ravel_conditions(values, datum_P, datum_T)
        if occupancies is None:
            occupancies = [None] * len(configurations)

        # calculate the points arrays, should be 2d array of points arrays
        points = np.array([calculate_points_array(phase_constituents, config, occup) for config, occup in zip(configurations, occupancies)])

        # add everything to the calculate_dict
        calculate_dict['P'] = np.concatenate([calculate_dict['P'], P])
        calculate_dict['T'] = np.concatenate([calculate_dict['T'], T])
        calculate_dict['points'] = np.concatenate([calculate_dict['points'], np.repeat(points, len(T)/points.shape[0], axis=0)], axis=0)
        calculate_dict['values'] = np.concatenate([calculate_dict['values'], values.flatten()])
        calculate_dict['weights'].extend(np.array([datum.get('weight', 1.0) for _ in range(values.flatten().size)]).flatten())
        calculate_dict['references'].extend([datum.get('reference', "") for _ in range(values.flatten().size)])

    return calculate_dict


def get_thermochemical_data(dbf, comps, phases, datasets, weight_dict=None, symbols_to_fit=None):
    """

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    phases : list
        List of phases to consider
    datasets : espei.utils.PickleableTinyDB
        Datasets that contain single phase data
    weight_dict : dict
        Dictionary of weights for each data type, e.g. {'HM': 200, 'SM': 2}
    symbols_to_fit : list
        Parameters to fit. Used to build the models and PhaseRecords.

    Returns
    -------
    list
        List of data dictionaries to iterate over
    """
    # phase by phase, then property by property, then by model exclusions
    if weight_dict is None:
        weight_dict = {}

    if symbols_to_fit is not None:
        symbols_to_fit = sorted(symbols_to_fit)
    else:
        symbols_to_fit = database_symbols_to_fit(dbf)

    # estimated from NIST TRC uncertainties
    property_std_deviation = {
        'HM': 500.0/weight_dict.get('HM', 1.0),  # J/mol
        'SM':   0.2/weight_dict.get('SM', 1.0),  # J/K-mol
        'CPM':  0.2/weight_dict.get('CPM', 1.0),  # J/K-mol
    }
    properties = ['HM_FORM', 'SM_FORM', 'CPM_FORM', 'HM_MIX', 'SM_MIX', 'CPM_MIX']

    ref_states = []
    for el in get_pure_elements(dbf, comps):
        ref_state = ReferenceState(el, dbf.refstates[el]['phase'])
        ref_states.append(ref_state)
    all_data_dicts = []
    for phase_name in phases:
        for prop in properties:
            desired_data = get_prop_data(comps, phase_name, prop, datasets, additional_query=(where('solver').exists()))
            if len(desired_data) == 0:
                continue
            unique_exclusions = set([tuple(sorted(d.get('excluded_model_contributions', []))) for d in desired_data])
            for exclusion in unique_exclusions:
                data_dict = {
                    'phase_name': phase_name,
                    'prop': prop,
                    # needs the following keys to be added:
                    # species, calculate_dict, phase_records, model, output, weights
                }
                # get all the data with these model exclusions
                if exclusion == tuple([]):
                    exc_search = (~where('excluded_model_contributions').exists()) & (where('solver').exists())
                else:
                    exc_search = (where('excluded_model_contributions').test(lambda x: tuple(sorted(x)) == exclusion)) & (where('solver').exists())
                curr_data = get_prop_data(comps, phase_name, prop, datasets, additional_query=exc_search)
                calculate_dict = get_prop_samples(dbf, comps, phase_name, curr_data)
                mod = Model(dbf, comps, phase_name, parameters=symbols_to_fit)
                if prop.endswith('_FORM'):
                    output = ''.join(prop.split('_')[:-1])+'R'
                    mod.shift_reference_state(ref_states, dbf, contrib_mods={e: sympy.S.Zero for e in exclusion})
                else:
                    output = prop
                for contrib in exclusion:
                    mod.models[contrib] = sympy.S.Zero
                    mod.reference_model.models[contrib] = sympy.S.Zero
                species = sorted(unpack_components(dbf, comps), key=str)
                data_dict['species'] = species
                model = {phase_name: mod}
                statevar_dict = {getattr(v, c, None): vals for c, vals in calculate_dict.items() if isinstance(getattr(v, c, None), v.StateVariable)}
                statevar_dict = OrderedDict(sorted(statevar_dict.items(), key=lambda x: str(x[0])))
                str_statevar_dict = OrderedDict((str(k), vals) for k, vals in statevar_dict.items())
                phase_records = build_phase_records(dbf, species, [phase_name], statevar_dict, model,
                                                    output=output, parameters={s: 0 for s in symbols_to_fit},
                                                    build_gradients=False, build_hessians=False)
                data_dict['str_statevar_dict'] = str_statevar_dict
                data_dict['phase_records'] = phase_records
                data_dict['calculate_dict'] = calculate_dict
                data_dict['model'] = model
                data_dict['output'] = output
                data_dict['weights'] = np.array(property_std_deviation[prop.split('_')[0]])/np.array(calculate_dict.pop('weights'))
                all_data_dicts.append(data_dict)
    return all_data_dicts


def calculate_non_equilibrium_thermochemical_probability(dbf, thermochemical_data, parameters=None):
    """
    Calculate the weighted single phase error in the Database

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    thermochemical_data : list
        List of thermochemical data dicts
    parameters : np.ndarray
        Array of parameters to calculate the error with.

    Returns
    -------
    float
        A single float of the residual sum of square errors

    Notes
    -----
    There are different single phase values, HM_MIX, SM_FORM, CP_FORM, etc.
    Each of these have different units and the error cannot be compared directly.
    To normalize all of the errors, a normalization factor must be used.
    Equation 2.59 and 2.60 in Lukas, Fries, and Sundman "Computational Thermodynamics" shows how this can be considered.
    Each type of error will be weighted by the reciprocal of the estimated uncertainty in the measured value and conditions.
    The weighting factor is calculated by
    $ p_i = (\Delta L_i)^{-1} $
    where $ \Delta L_i $ is the uncertainty in the measurement.
    We will neglect the uncertainty for quantities such as temperature, assuming they are small.

    """
    if parameters is None:
        parameters = np.array([])

    prob_error = 0.0
    for data in thermochemical_data:
        phase_name = data['phase_name']
        output = data['output']
        phase_records = data['phase_records']
        sample_values = data['calculate_dict']['values']

        update_phase_record_parameters(phase_records, parameters)
        results = calculate_(dbf, data['species'], [phase_name],
                             data['str_statevar_dict'], data['model'],
                             phase_records, output=output, broadcast=False,
                             points=data['calculate_dict']['points'])[output]
        differences = results - sample_values
        probabilities = norm.logpdf(differences, loc=0, scale=data['weights'])
        prob_sum = np.sum(probabilities)
        logging.debug(f"Thermochemical error - {data['prop']}({phase_name}) = {prob_sum} - data: {sample_values}, differences: {differences}, probabilities: {probabilities}, references: {data['calculate_dict']['references']}")
        prob_error += prob_sum
    return prob_error

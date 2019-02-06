"""
Calculate driving_force due to ZPF tielines.

The general approach is similar to the PanOptimizer rough search method.

1. With all phases active, calculate the chemical potentials of the tieline
   endpoints via ``equilibrium`` calls. Done in ``estimate_hyperplane``.
2. Calculate the target chemical potentials, which are the average chemical
   potentials of all of the current chemical potentials at the tieline endpoints.
3. Calculate the current chemical potentials of the desired single phases
4. The error is the difference between these chemical potentials

There's some special handling for tieline endpoints where we do not know the
composition conditions to calculate chemical potentials at.
"""

import operator, logging
from copy import deepcopy
from collections import defaultdict, OrderedDict

import numpy as np
from scipy.stats import norm
import tinydb

from pycalphad import calculate, equilibrium, variables as v

TRACE = 15  # TRACE logging level

def _safe_index(items, index):
    try:
        return items[index]
    except IndexError:
        return None


def _mix_compositions(cond_key, this_comp, curr_idx, other_compositions, mix_fraction=0.001):
    for idx, other_conds in enumerate(other_compositions):
        if idx != curr_idx:
            other_comp = other_conds[cond_key]
            if not np.isnan(np.array(other_comp, dtype=np.float64)):
                return this_comp*(1-mix_fraction) + other_comp*mix_fraction
    return this_comp

def _adjust_compositions(comp_dicts):
    """
    Adjust compositions of stoichiometric phases to be slightly off stoichiometry in the direction of the corresponding tieline point

    Parameters
    ----------
    comp_dicts : list
        Each comp dict is a tuple of ({composition condition dict}, phase_flags)

    Returns
    -------
    list
        Adjusted compositions
    """
    new_comp_dicts = []
    other_compositions = [conds for conds, flag in comp_dicts]

    for idx, (conds, phase_flag) in enumerate(comp_dicts):
        new_conds = {}
        for comp_var, composition in conds.items():
            if not np.isnan(np.array(composition, dtype=np.float64)):
                # this composition must be adjusted
                # find the first non-nan composition that isn't
                # this index and mix the two with 99.9% this composition
                new_conds[comp_var] = _mix_compositions(comp_var, composition, idx, other_compositions)
            else:
                new_conds[comp_var] = composition
        new_comp_dicts.append((new_conds, phase_flag))
    return new_comp_dicts

def phase_is_stoichiometric(dbf, phase_name, species):
    """
    Return True if phase has no internal degrees of freedom

    Parameters
    ----------
    dbf : Database
    phase_name : str
    species : set

    Returns
    -------
    bool

    """
    return not any((len(species.intersection(subl)) > 1 for subl in dbf.phases[phase_name].constituents))

def get_zpf_data(dbf, comps, phases, datasets, adjust_stoichometric=True):
    """
    Return the ZPF data used in the calculation of ZPF error

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
    adjust_stoichometric : bool
        If True, any regions with all stoichiometric phases will create hyperplane
        composition conditions that are slightly perturbed from the stoichiometric value.

    Returns
    -------
    list
        List of data dictionaries with keys ``weight``, ``data_comps`` and
        ``phase_regions``. ``data_comps`` are the components for the data in
        question. ``phase_regions`` are the ZPF phases, state variables and compositions.

    """
    desired_data = datasets.search((tinydb.where('output') == 'ZPF') &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))

    zpf_data = []
    for data in desired_data:
        payload = data['values']
        conditions = data['conditions']
        species = {v.Species(c) for c in data['components']}
        stoichiometric_phases = {ph for ph in phases if phase_is_stoichiometric(dbf, ph, species)}
        # create a dictionary of each set of phases containing a list of individual points on the tieline
        # individual tieline points are tuples of (conditions, {composition dictionaries})
        phase_regions = defaultdict(lambda: list())
        # TODO: Fix to only include equilibria listed in 'phases'
        for idx, p in enumerate(payload):
            phase_key = tuple(sorted(rp[0] for rp in p))
            if len(phase_key) < 2:
                # Skip single-phase regions for fitting purposes
                continue
            # Need to sort 'p' here so we have the sorted ordering used in 'phase_key'
            # rp[3] optionally contains additional flags, e.g., "disordered", to help the solver
            # each comp dict is a tuple of ({composition condition dict}, phase_flags)
            comp_dicts = [(dict(zip([v.X(x.upper()) for x in rp[1]], rp[2])), _safe_index(rp, 3))
                          for rp in sorted(p, key=operator.itemgetter(0))]
            if len(set(phase_key).difference(stoichiometric_phases)) == 0 and adjust_stoichometric:
                # all phases are stoichiometric, adjusting composition
                hyperplane_comp_dicts = _adjust_compositions(comp_dicts)
                logging.log(TRACE, 'All phases stoichiometric. Adjusting compositions from {} to {}'.format(comp_dicts, hyperplane_comp_dicts))
            else:
                hyperplane_comp_dicts = deepcopy(comp_dicts)

            cur_conds = {}
            for key, value in conditions.items():
                value = np.atleast_1d(np.asarray(value))
                if len(value) > 1:
                    value = value[idx]
                cur_conds[getattr(v, key)] = float(value)
            phase_regions[phase_key].append((cur_conds, comp_dicts, hyperplane_comp_dicts))

        data_dict = {
            'weight': data.get('weight', 1.0),
            'data_comps': list(set(data['components']).union({'VA'})),
            'phase_regions': phase_regions,
            'dataset_reference': data['reference']
        }
        zpf_data.append(data_dict)
    return zpf_data


def estimate_hyperplane(dbf, comps, phases, current_statevars, comp_dicts, phase_models, parameters,
                        callables=None):
    """
    Calculate the chemical potentials for the target hyperplane, one vertex at a time

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    phases : list
        List of phases to consider
    current_statevars : dict
        Dictionary of state variables, e.g. v.P and v.T, no compositions.
    comp_dicts : list
        List of tuples of composition dictionaries and phase flags. Composition
        dictionaries are pycalphad variable dicts and the flag is a string e.g.
        ({v.X('CU'): 0.5}, 'disordered')
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    callables : dict
        Callables to pass to pycalphad

    Returns
    -------
    numpy.ndarray
        Array of chemical potentials.

    Notes
    -----
    This takes just *one* set of phase equilibria, e.g. a dataset point of
    [['FCC_A1', ['CU'], [0.1]], ['LAVES_C15', ['CU'], [0.3]]]
    and calculates the chemical potentials given all the phases possible at the
    given compositions. Then the average chemical potentials of each end point
    are taken as the target hyperplane for the given equilibria.

    """
    target_hyperplane_chempots = []
    parameters = OrderedDict(sorted(parameters.items(), key=str))
    # TODO: unclear whether we use phase_flag and how it would be used. It should be just a 'disordered' kind of flag.
    for cond_dict, phase_flag in comp_dicts:
        # We are now considering a particular tie vertex
        for key, val in cond_dict.items():
            if val is None:
                cond_dict[key] = np.nan
        cond_dict.update(current_statevars)
        if np.any(np.isnan(list(cond_dict.values()))):
            # This composition is unknown -- it doesn't contribute to hyperplane estimation
            pass
        else:
            # Extract chemical potential hyperplane from multi-phase calculation
            # Note that we consider all phases in the system, not just ones in this tie region
            multi_eqdata = equilibrium(dbf, comps, phases, cond_dict, model=phase_models,
                                       parameters=parameters, callables=callables,)
            # Does there exist only a single phase in the result with zero internal degrees of freedom?
            # We should exclude those chemical potentials from the average because they are meaningless.
            num_phases = np.sum(multi_eqdata['Phase'].values.squeeze() != '')
            Y_values = multi_eqdata.Y.values.squeeze()
            no_internal_dof = np.all((np.isclose(Y_values, 1.)) | np.isnan(Y_values))
            MU_values = multi_eqdata['MU'].values.squeeze()
            if (num_phases == 1) and no_internal_dof:
                target_hyperplane_chempots.append(np.full_like(MU_values, np.nan))
            else:
                target_hyperplane_chempots.append(MU_values)
    target_hyperplane_chempots = np.nanmean(target_hyperplane_chempots, axis=0, dtype=np.float)
    return target_hyperplane_chempots


def driving_force_to_hyperplane(dbf, comps, current_phase, cond_dict, target_hyperplane_chempots,
                        phase_flag, phase_models, parameters, callables=None):
    """Calculate the integrated driving force between the current hyperplane and target hyperplane.

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    current_phase : list
        List of phases to consider
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    callables : dict
        Callables to pass to pycalphad
    cond_dict : dict
            Dictionary of state variables, e.g. v.P and v.T, v.X
    target_hyperplane_chempots : numpy.ndarray
        Array of chemical potentials for target equilibrium hyperplane.
    phase_flag : str
        String of phase flag, e.g. 'disordered'.
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium

    Returns
    -------
    float
        Single value for the total error between the current hyperplane and target hyperplane.

    """
    if np.any(np.isnan(list(cond_dict.values()))):
        # We don't actually know the phase composition here, so we estimate it
        single_eqdata = calculate(dbf, comps, [current_phase],
                                  T=cond_dict[v.T], P=cond_dict[v.P],
                                  model=phase_models, parameters=parameters, pdens=100,
                                  callables=callables)
        driving_force = np.multiply(target_hyperplane_chempots, single_eqdata['X'].values).sum(axis=-1) - single_eqdata['GM'].values
        driving_force = float(driving_force.max())
    elif phase_flag == 'disordered':
        # Construct disordered sublattice configuration from composition dict
        # Compute energy
        # Compute residual driving force
        # TODO: Check that it actually makes sense to declare this phase 'disordered'
        num_dof = sum([len(set(c).intersection(comps)) for c in dbf.phases[current_phase].constituents])
        desired_sitefracs = np.ones(num_dof, dtype=np.float)
        dof_idx = 0
        for c in dbf.phases[current_phase].constituents:
            dof = sorted(set(c).intersection(comps))
            if (len(dof) == 1) and (dof[0] == 'VA'):
                return 0
            # If it's disordered config of BCC_B2 with VA, disordered config is tiny vacancy count
            sitefracs_to_add = np.array([cond_dict.get(v.X(d)) for d in dof], dtype=np.float)
            # Fix composition of dependent component
            sitefracs_to_add[np.isnan(sitefracs_to_add)] = 1 - np.nansum(sitefracs_to_add)
            desired_sitefracs[dof_idx:dof_idx + len(dof)] = sitefracs_to_add
            dof_idx += len(dof)
        single_eqdata = calculate(dbf, comps, [current_phase], T=cond_dict[v.T],
                                  P=cond_dict[v.P], points=desired_sitefracs,
                                  model=phase_models, parameters=parameters, callables=callables,)
        driving_force = np.multiply(target_hyperplane_chempots, single_eqdata['X'].values).sum(axis=-1) - single_eqdata['GM'].values
        driving_force = float(np.squeeze(driving_force))
    else:
        # Extract energies from single-phase calculations
        single_eqdata = equilibrium(dbf, comps, [current_phase], cond_dict, model=phase_models,
                                    parameters=parameters, callables=callables)
        if np.all(np.isnan(single_eqdata['NP'].values)):
            logging.debug('Calculation failure: all NaN phases with phases: {}, conditions: {}, parameters {}'.format(current_phase, cond_dict, parameters))
            return np.inf
        select_energy = float(single_eqdata['GM'].values)
        region_comps = []
        for comp in [c for c in sorted(comps) if c != 'VA']:
            region_comps.append(cond_dict.get(v.X(comp), np.nan))
        region_comps[region_comps.index(np.nan)] = 1 - np.nansum(region_comps)
        driving_force = np.multiply(target_hyperplane_chempots, region_comps).sum() - select_energy
        driving_force = float(driving_force)
    return driving_force


def calculate_zpf_error(dbf, comps, phases, datasets, phase_models, parameters=None, callables=None, data_weight=1.0):
    """
    Calculate error due to phase equilibria data

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
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    callables : dict
        Callables to pass to pycalphad
    data_weight : float
        Scaling factor for the standard deviation of the measurement of a
        tieline which has units J/mol. The standard deviation is 1000 J/mol
        and the scaling factor defaults to 1.0.

    Returns
    -------
    float
        Log probability of ZPF error

    Notes
    -----
    The physical picture of the standard deviation is that we've measured a ZPF
    line. That line corresponds to some equilibrium chemical potentials. The
    standard deviation is the standard deviation of those 'measured' chemical
    potentials.

    """
    prob_error = 0.0
    for data in get_zpf_data(dbf, comps, phases, datasets):
        phase_regions = data['phase_regions']
        data_comps = data['data_comps']
        weight = data['weight']
        dataset_ref = data['dataset_reference']
        # for each set of phases in equilibrium and their individual tieline points
        for region, region_eq in phase_regions.items():
            # for each tieline region conditions and compositions
            for current_statevars, comp_dicts, hyperplane_comp_dicts in region_eq:
                # a "region" is a set of phase equilibria
                eq_str = "conds: ({}), comps: ({})".format(current_statevars, ', '.join(['{}: {}'.format(ph,c[0]) for ph, c in zip(region, comp_dicts)]))
                target_hyperplane = estimate_hyperplane(dbf, data_comps, phases, current_statevars, hyperplane_comp_dicts, phase_models, parameters, callables=callables)
                if np.any(np.isnan(target_hyperplane)):
                    logging.warning('Found a NaN ZPF driving force. Equilibria: ({}), reference: {}. Target hyperplane: {}. If this data point consistently gives NaN, consider removing it.'.format(eq_str, dataset_ref, target_hyperplane))
                # Now perform the equilibrium calculation for the isolated phases and add the result to the error record
                for current_phase, cond_dict in zip(region, comp_dicts):
                    # TODO: Messy unpacking
                    cond_dict, phase_flag = cond_dict
                    # We are now considering a particular tie vertex
                    for key, val in cond_dict.items():
                        if val is None:
                            cond_dict[key] = np.nan
                    cond_dict.update(current_statevars)
                    driving_force = driving_force_to_hyperplane(dbf, data_comps, current_phase, cond_dict, target_hyperplane,
                                                  phase_flag, phase_models, parameters, callables=callables)
                    vertex_prob = norm(loc=0, scale=1000/data_weight/weight).logpdf(driving_force)
                    prob_error += vertex_prob
                    logging.debug('ZPF error - Equilibria: ({}), current phase: {}, driving force: {}, probability: {}, reference: {}'.format(eq_str, current_phase, driving_force, vertex_prob, dataset_ref))
    if np.isnan(prob_error):
        return -np.inf
    return prob_error


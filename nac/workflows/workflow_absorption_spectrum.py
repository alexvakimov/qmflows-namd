
__all__ = ['workflow_oscillator_strength']

from itertools import chain
from noodles import (gather, schedule)
from nac.common import (
    Matrix, Vector, change_mol_units, compute_center_of_mass, h2ev,
    retrieve_hdf5_data)
from nac.integrals.multipole_matrices import get_multipole_matrix
from nac.schedule.components import calculate_mos
from nac.workflows.initialization import initialize
from qmflows import run
from qmflows.parsers import parse_string_xyz
from scipy.constants import physical_constants

import logging
import numpy as np

# Type hints
from typing import (Any, Dict, List, Tuple)

# Get logger
logger = logging.getLogger(__name__)

# Planck con`stant in ev . s
hbar_evs = physical_constants['Planck constant over 2 pi in eV s'][0]


def workflow_oscillator_strength(workflow_settings: Dict):
    """
    Compute the oscillator strength.

    :param workflow_settings: Arguments to compute the oscillators see:
    `data/schemas/absorption_spectrum.json
    :returns: None
    """
    # Arguments to compute the orbitals and configure the workflow. see:
    # `data/schemas/general_settings.json
    config = workflow_settings['general_settings']

    # Dictionary containing the general configuration
    config.update(initialize(**config))

    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(**config)

    # geometries in atomic units
    molecules_au = [change_mol_units(parse_string_xyz(gs))
                    for gs in config['geometries']]

    # Construct initial and final states ranges
    transition_args = [workflow_settings[key] for key in
                       ['initial_states', 'final_states', 'nHOMO']]
    initial_states, final_states = build_transitions(*transition_args)

    # Make a promise object the function the compute the Oscillator Strenghts
    scheduleOscillator = schedule(calc_oscillator_strenghts)

    oscillators = gather(
        *[scheduleOscillator(
            i, mol,  mo_paths_hdf5, initial_states, final_states,
            config) for i, mol in enumerate(molecules_au)
          if i % workflow_settings['calculate_oscillator_every'] == 0])

    energies, promised_cross_section = create_promised_cross_section(
        oscillators, workflow_settings['broadening'], workflow_settings['energy_range'],
        workflow_settings['convolution'], workflow_settings['calculate_oscillator_every'])

    cross_section, data = run(
        gather(promised_cross_section, oscillators), folder=config['work_dir'])

    return store_data(data, energies, cross_section)


def store_data(data: Tuple, energies: Vector, cross_section: Vector):
    """
    Print the data into tables.
    """
    # Transform the energy to nm^-1
    energies_nm = energies * 1240

    # Save cross section
    np.savetxt('cross_section_cm.txt',
               np.stack((energies, energies_nm, cross_section, cross_section * 1e16), axis=1),
               header='Energy[eV] Energy[nm^-1] photoabsorption_cross_section[cm^2] photoabsorption_cross_section[^2]')

    # molar extinction coefficients (e in M-1 cm-1)
    nA = physical_constants['Avogadro constant'][0]
    cte = np.log(10) * 1e3 / nA
    extinction_coefficients = cross_section / cte
    np.savetxt('molar_extinction_coefficients.txt',
               np.stack((energies, energies_nm, extinction_coefficients), axis=1),
               header='Energy[eV] Energy[nm^-1] Extinction_coefficients[M^-1 cm^-1]')

    print("Calculation Done")

    # Write data in human readable format
    write_information(data)

    return data


def create_promised_cross_section(
        oscillators: List, broadening: float, energy_range: Tuple,
        convolution: str, calculate_oscillator_every: int):
    """
    Create the function call that schedule the computation of the
    photoabsorption cross section
    """
    # Energy grid in  hartrees
    initial_energy = energy_range[0]
    final_energy = energy_range[1]
    npoints = 10 * (final_energy - initial_energy) // broadening
    energies = np.linspace(initial_energy, final_energy, int(npoints))
    # Compute the cross section

    schedule_cross_section = schedule(compute_cross_section_grid)

    cross_section = schedule_cross_section(
        oscillators, convolution, energies, broadening,
        calculate_oscillator_every)

    return energies, cross_section


def compute_cross_section_grid(
        oscillators: List, convolution: str, energies: Vector,
        broadening: float, calculate_oscillator_every: int) -> float:
    """
    Compute the photoabsorption cross section as a function of the energy.
    See: The UV absorption of nucleobases: semi-classical ab initio spectra
    simulations. Phys. Chem. Chem. Phys., 2010, 12, 4959–4967
    """
    # speed of light in m s^-1
    c = physical_constants['speed of light in vacuum'][0]
    # Mass of the electron in Kg
    m = physical_constants['electron mass'][0]
    # Charge of the electron in C
    e = physical_constants['elementary charge'][0]
    # Vacuum permitivity in C^2 N^-1 m^-2
    e0 = 8.854187817620e-12

    # Constant in cm^2
    cte = np.pi * (e ** 2) / (2 * m * c * e0) * 1e4  # m^2 to cm^2

    # convulation functions for the intensity
    convolution_functions = {'gaussian': gaussian_distribution,
                             'lorentzian': lorentzian_distribution}
    fun_convolution = convolution_functions[convolution]

    def compute_cross_section(energy: float) -> Vector:
        """
        compute a single value of the photoabsorption cross section by
        rearranging oscillator strengths by initial states and perform
        the summation.
        """
        # Photo absorption in length
        grid_ev = cte * sum(
            sum(
                sum(osc['fij'] * fun_convolution(energy, osc['deltaE'] * h2ev, broadening)
                    for osc in ws) / len(ws)
                for ws in zip(*arr)) for arr in zip(*oscillators))

        # convert the cross section to cm^2

        return grid_ev

    vectorized_cross_section = np.vectorize(compute_cross_section)

    return vectorized_cross_section(energies)


def gaussian_distribution(x: float, center: float, delta: float) -> Vector:
    """
    Return gaussian as described at:
    Phys. Chem. Chem. Phys., 2010, 12, 4959–4967
    """
    pre_expo = np.sqrt(2 / np.pi) * (hbar_evs / delta)
    expo = np.exp(-2 * ((x - center) / delta) ** 2)

    return pre_expo * expo


def lorentzian_distribution(
        x: float, center: float, delta: float) -> Vector:
    """
    Return a Lorentzian as described at:
    Phys. Chem. Chem. Phys., 2010, 12, 4959–4967
    """
    cte = (hbar_evs * delta) / (2 * np.pi)
    denominator = (x - center) ** 2 + (delta / 2) ** 2

    return cte * (1 / denominator)


def calc_oscillator_strenghts(
        i: int, atoms: List, mo_paths_hdf5: str, initial_states: Vector, final_states: Matrix,
        config: Dict):
    """
    Use the Molecular orbital Energies and Coefficients to compute the
    oscillator_strength.

    :param i: time frame
    :param atoms: Molecular geometry.
    :type atoms: [namedtuple("AtomXYZ", ("symbol", "xyz"))]
    :param initial_states: List of the initial Electronic states.
    :type initial_states: [Int]
    :param final_states: List containing the sets of possible electronic
    states.
    :type final_states: [[Int]]
    :param config: Configuration to perform the calculations
    """
    path_hdf5 = config['path_hdf5']
    # Energy and coefficients at time t
    es, coeffs = retrieve_hdf5_data(path_hdf5, mo_paths_hdf5[i])

    logger.info("Computing the oscillator strength at time: {}".format(i))
    # Overlap matrix

    # Dipole matrix in sphericals
    mtx_integrals_spher = get_multipole_matrix(i, atoms, config, 'dipole')
    # Origin of the dipole
    rc = compute_center_of_mass(atoms)

    oscillators = [
        compute_oscillator_strength(
            rc, atoms, config['dictCGFs'], es, coeffs, mtx_integrals_spher, initialS, fs)
        for initialS, fs in zip(initial_states, final_states)]
    return oscillators


def compute_oscillator_strength(
        rc: Tuple, atoms: List, dictCGFs: Dict, es: Vector, coeffs: Matrix,
        mtx_integrals_spher: Matrix, initialS: int, fs: List):
    """
    Compute the oscillator strenght using the matrix elements of the position
    operator:

    .. math:
    f_i->j = 2/3 * E_i->j * ∑^3_u=1 [ <ψi | r_u | ψj> ]^2

    where Ei→j is the single particle energy difference of the transition
    from the Kohn-Sham state ψi to state ψj and rμ = x,y,z is the position
    operator.
    """
    # Retrieve the molecular orbital coefficients and energies
    css_i = coeffs[:, initialS]
    energy_i = es[initialS]

    # Compute the oscillator strength
    xs = []
    for finalS in fs:
        # Get the molecular orbitals coefficients and energies
        css_j = coeffs[:, finalS]
        energy_j = es[finalS]
        deltaE = energy_j - energy_i

        # compute the oscillator strength and the transition dipole components
        fij, components = oscillator_strength(
            css_i, css_j, deltaE, mtx_integrals_spher)

        st = 'transition {:d} -> {:d} Fij = {:f}\n'.format(
            initialS, finalS, fij)
        logger.info(st)
        osc = {'initialS': initialS, 'finalS': finalS, 'energy_i': energy_i,
               'energy_j': energy_j, 'deltaE': deltaE, 'fij': fij, 'components': components}
        xs.append(osc)

    return xs


def write_information(data: Tuple) -> None:
    """
    Write to a file the oscillator strenght information
    """
    header = "Transition initial_state[eV] final_state[eV] delta_E[eV] delta_E[nm^-1] fij Transition_dipole_components [a.u.]\n"
    filename = 'oscillators.txt'
    with open(filename, 'w') as f:
        f.write(header)
    for xs in list(chain(*data)):
        for args in xs:
            write_oscillator(filename, args)


def write_oscillator(
        filename: str, osc: Dict) -> None:
    """
    Write oscillator strenght information in one file
    """
    energy_ev = osc['deltaE'] * h2ev
    initial_ev = osc['energy_i'] * h2ev
    final_ev = osc['energy_j'] * h2ev
    energy_nm = 1240 / energy_ev
    fmt = '{} -> {} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} \
    {:11.5f} {:11.5f} {:11.5f}\n'.format(
        osc['initialS'] + 1, osc['finalS'] + 1, initial_ev, final_ev, energy_ev,
        energy_nm, osc['fij'], *osc['components'])

    with open(filename, 'a') as f:
        f.write(fmt)


def oscillator_strength(css_i: Matrix, css_j: Matrix, energy: float,
                        mtx_integrals_spher: Matrix) -> float:
    """
    Calculate the oscillator strength between two state i and j using a
    molecular geometry in atomic units, a set of contracted gauss functions
    normalized, the coefficients for both states, the nergy difference between
    the states and a matrix to transform from cartesian to spherical
    coordinates in case the coefficients are given in cartesian coordinates.

    :param css_i: MO coefficients of initial state
    :param css_j: MO coefficients of final state
    :param energy: energy difference i -> j.
    :param mtx_integrals_triang: matrix containing the dipole integrals
    :returns: Oscillator strength
    """
    components = tuple(
        map(lambda mtx: np.dot(css_i, np.dot(mtx, css_j)),
            mtx_integrals_spher))

    sum_integrals = sum(x ** 2 for x in components)

    fij = (2 / 3) * energy * sum_integrals

    return fij, components


def build_transitions(
        initial_states: Any, final_states: Any, nHOMO: int) -> Tuple:
    """
    Build the set of initial state to compute the oscillator strengths with.
    If the user provided two integers a range of transition is built assuming
    that those numbers are the lowest ang highest orbitals to use in the space.
    Otherwise it is assumed that the user provided the range of initial and
    final orbitals.
    """
    if all(isinstance(s, list) for s in [initial_states, final_states]):
        # Shift the range 1 to start the index at 0
        initial = np.array(initial_states) - 1
        final = np.array(final_states) - 1
    elif all(isinstance(s, int) for s in [initial_states, final_states]):
        # Create the range of initial and final states using the lowest
        # and highest orbital provided by the user
        initial = np.arange(initial_states - 1, nHOMO)
        range_final = np.arange(nHOMO, final_states)
        dim_x = initial.size
        dim_y = range_final.size
        # Matrix of final states
        final = np.tile(range_final, dim_x).reshape(dim_x, dim_y)
    else:
        raise RuntimeError('I did not understand the initial and final state format')

    return initial, final

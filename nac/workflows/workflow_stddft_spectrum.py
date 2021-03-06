__all__ = ['workflow_stddft']

from nac.common import (
    Matrix, Vector, change_mol_units, h2ev, hardness, retrieve_hdf5_data, xc)
from nac.integrals.multipole_matrices import get_multipole_matrix
from nac.integrals.spherical_Cartesian_cgf import (calc_orbital_Slabels, read_basis_format)
from nac.schedule.components import calculate_mos
from nac.workflows.initialization import initialize
from qmflows.parsers import parse_string_xyz
from qmflows import run
from noodles import (gather, schedule)
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from typing import (Dict, List)
import h5py
import numpy as np
import os


def workflow_stddft(workflow_settings: Dict):
    """
    Compute the excited states using simplified TDDFT

    :param workflow_settings: Arguments to compute the oscillators see:
    `data/schemas/absorption_spectrum.json
    :returns: None
    """
    # Arguments to compute the orbitals and configure the workflow. see:
    # `data/schemas/general_settings.json
    config = workflow_settings['general_settings']

    # Dictionary containing the general configuration
    config.update(initialize(**config))

    # Single Point calculations settings using CP2K
    mo_paths_hdf5 = calculate_mos(**config)

    # Read structures
    molecules_au = [change_mol_units(parse_string_xyz(gs))
                    for gs in config['geometries']]

    # Noodles promised call
    scheduleTDDFT = schedule(compute_excited_states_tddft)

    results = gather(
       *[scheduleTDDFT(
           i, mol, mo_paths_hdf5, workflow_settings['xc_dft'],
           workflow_settings['ci_range'], workflow_settings['nHOMO'],
           workflow_settings['tddft'], config)
         for i, mol in enumerate(molecules_au)
         if i % workflow_settings['calculate_oscillator_every'] == 0])

    return run(results, folder=config['work_dir'])


def compute_excited_states_tddft(
           i: int, mol: List, mo_paths_hdf5, xc_dft: str, ci_range: list,
        nocc: int, tddft: str, config: Dict):
    """
    Compute the excited states properties (energy and coefficients) for a given
    `ci_range` using the `tddft` method and `xc_dft` exchange functional.
    """
    print("Reading energies and mo coefficients")
    energy, c_ao = retrieve_hdf5_data(config['path_hdf5'], mo_paths_hdf5[i])

    # Number of virtual orbitals
    nvirt = c_ao.shape[1] - nocc

    if tddft == 'sing_orb':
        omega = -np.subtract(
            energy[:nocc].reshape(nocc, 1), energy[nocc:].reshape(nvirt, 1).T).reshape(nocc*nvirt)
        xia = np.eye(nocc*nvirt)
    else:
        # Call the function that computes overlaps
        print("Reading or computing the overlap matrix")
        multipoles = get_multipole_matrix(
            i, mol, config, 'overlap')

        # Make a function tha returns in transition density charges
        print("Computing the transition density charges")
        q = transition_density_charges(mol, config, multipoles, c_ao)

        # Make a function that compute the Mataga-Nishimoto-Ohno_Klopman
        # damped Columb and Excgange law functions
        print("Computing the gamma functions for Exchange and Coulomb integrals")
        gamma_J, gamma_K = compute_MNOK_integrals(mol, xc_dft)

        # Compute the Couloumb and Exchange integrals
        # If xc_dft is a pure functional, ax=0, thus the pqrs_J ints are not needed
        # and can be set to 0
        print("Computing the Exchange and Coulomb integrals")
        if (xc(xc_dft)['type'] == 'pure'):
            size = energy.size
            pqrs_J = np.zeros((size, size, size, size))
        else:
            pqrs_J = np.tensordot(q, np.tensordot(q, gamma_J, axes=(0, 1)), axes=(0, 2))
        pqrs_K = np.tensordot(q, np.tensordot(q, gamma_K, axes=(0, 1)), axes=(0, 2))

        # Construct the Tamm-Dancoff matrix A for each pair of i->a transition
        print("Constructing the A matrix for TDDFT calculation")
        a_mat = construct_A_matrix_tddft(pqrs_J, pqrs_K, nocc, nvirt, xc_dft, energy)

        if tddft == 'stddft':
            print('sTDDFT has not been implemented yet !')
            # Solve the eigenvalue problem = A * cis = omega * cis
        elif tddft == 'stda':
            print("This is a TDA calculation ! \n Solving the eigenvalue problem")
            omega, xia = np.linalg.eig(a_mat)
        else:
            msg = "Only the stda method is available"
            raise RuntimeError(msg)

    return compute_oscillator_strengths(
        i, mol, tddft, config, energy, c_ao, multipoles, nocc, nvirt, omega, xia)


def compute_oscillator_strengths(
        i: int, mol: object, tddft: str, config: Dict, energy: Vector, c_ao: Matrix,
        multipoles: Matrix, nocc: int, nvirt: int, omega: Vector, xia: Matrix):
    """
    Compute oscillator strengths
    The formula can be rearranged like this:
    f_I = 2/3 * np.sqrt(2 * omega_I) * sum_ia ( np.sqrt(e_diff_ia) * xia * tdm_x) ** 2 + y^2 + z^2

    :param i: index
    :param mol: molecular geometry
    :param tddft: type of calculation
    :param config: Setting for the current calculation
    :param energy: energy of the orbitals
    :param c_ao: coefficients of the molecular orbitals
    :param nocc: number of occupied orbitals
    :param nvirt: number of virtual orbitals
    :param omega: Omega parameter
    """
    # 1) Get the energy matrix i->a. Size: Nocc * Nvirt
    delta_ia = -np.subtract(
        energy[:nocc].reshape(nocc, 1), energy[nocc:].reshape(nvirt, 1).T).reshape(nocc*nvirt)

    def compute_transition_matrix(matrix):
        return np.stack(
            np.sum(
                np.sqrt(2 * delta_ia / omega[i]) * xia[:, i] * matrix)
            for i in range(nocc*nvirt))

    # 2) Compute the transition dipole matrix TDM(i->a)
    # Call the function that computes transition dipole moments integrals
    print("Reading or computing the transition dipole matrix")
    tdm = get_multipole_matrix(i, mol, config, 'dipole')

    def compute_tdmatrix(k):
        return np.linalg.multi_dot(
            [c_ao[:, :nocc].T, tdm[k, :, :], c_ao[:, nocc:]]).reshape(nocc*nvirt)

    td_matrices = (compute_tdmatrix(k) for k in range(3))

    # 3) Compute the transition dipole moments for each excited state i->a. Size: n_exc_states
    d_x, d_y, d_z = tuple(
        compute_transition_matrix(m) for m in td_matrices)

    # 4) Compute the oscillator strength
    f = 2 / 3 * omega * (d_x ** 2 + d_y ** 2 + d_z ** 2)

    # Write to output
    output = write_output_tddft(nocc, nvirt, omega, f, d_x, d_y, d_z, xia, energy)
    path_output = os.path.join(config['work_dir'], 'output_{}_{}.txt'.format(i, tddft))
    header = '{:^5s}{:^14s}{:^8s}{:^11s}{:^11s}{:^11s}{:^11s}{:<5s}{:^10s}{:<5s}{:^11s}{:^11s}'.format(
        'state', 'energy', 'f', 't_dip_x', 't_dip_y', 't_dip_y', 'weight', 'from', 'energy',
        'to', 'energy', 'delta_E')
    np.savetxt(path_output, output,
               fmt='%5d %10.3f %10.5f %10.5f %10.5f %10.5f %10.5f %3d %10.3f %3d %10.3f %10.3f',
               header=header)

    descriptors = False
    n_lowest = 3
    if descriptors:
        print("Reading or computing the quadrupole matrix")
        tqm = get_multipole_matrix(
            i, mol, config, 'quadrupole')

        descriptors = ex_descriptor(
            omega, f, xia, n_lowest, c_ao, multipoles, tdm, tqm, nocc, nvirt, mol, config)
        path_ex_output = os.path.join(config['work_dir'], 'descriptors_{}_{}.txt'.format(i, tddft))
        ex_header = '{:^5s}{:^14s}{:^10s}{:^10s}{:^12s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}'.format(
            'state', 'd_exc', 'd_exc_app', 'd_he', 'sigma_h', 'sigma_e', 'r_eh', 'bind_en',
            'energy', 'f')
        np.savetxt(
            path_ex_output, descriptors,
            fmt='%5d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f',
            header=ex_header)


def ex_descriptor(omega, f, xia, n_lowest, c_ao, s, tdm, tqm, nocc, nvirt, mol, config):
    """
    ADD DOCUMENTATION
    """
    # Reshape xia
    xia_I = xia.reshape(nocc, nvirt, nocc*nvirt)

    # Transform the transition density matrix into AO basis
    d0I_ao = np.stack(
        np.linalg.multi_dot(
            [c_ao[:, :nocc], xia_I[:, :, i], c_ao[:, nocc:].T]) for i in range(n_lowest))

    # Compute omega in excition analysis for the lowest n excitations
    om = get_omega(d0I_ao, s, n_lowest)

    # Compute the distribution of positions for the hole and electron
    xh, yh, zh = get_exciton_positions(d0I_ao, s, tdm, n_lowest, 'hole')
    xe, ye, ze = get_exciton_positions(d0I_ao, s, tdm, n_lowest, 'electron')

    # Compute the distribution of the square of position for the hole and electron
    x2h, y2h, z2h = get_exciton_positions(d0I_ao, s, tqm, n_lowest, 'hole')
    x2e, y2e, z2e = get_exciton_positions(d0I_ao, s, tqm, n_lowest, 'electron')

    # Compute the distribution of both hole and electron positions
    xhxe, yhye, zhze = get_exciton_positions(d0I_ao, s, tdm, n_lowest, 'both')

    # Compute Descriptors

    # Compute exciton size:
    d_exc = np.sqrt(
        ((x2h - 2 * xhxe + x2e) + (y2h - 2 * yhye + y2e) + (z2h - 2 * zhze + z2e)) / om)

    # Compute centroid electron_hole distance
    d_he = np.abs(((xe - xh) + (ye - yh) + (ze - zh)) / om)

    # Compute hole and electron size
    sigma_h = np.sqrt(
        (x2h / om - (xh / om) ** 2) + (y2h / om - (yh / om) ** 2) + (z2h / om - (zh / om) ** 2))
    sigma_e = np.sqrt(
        (x2e / om - (xe / om) ** 2) + (y2e / om - (ye / om) ** 2) + (z2e / om - (ze / om) ** 2))

    # Compute Pearson coefficients
    cov = (xhxe - xh * xe) + (yhye - yh * ye) + (zhze - zh * ze)
    r_eh = cov / (sigma_h * sigma_e)

    # Compute approximate d_exc and binding energy
    omega_ab = get_omega_ab(d0I_ao, s, n_lowest, mol, config)
    r_ab = get_r_ab(mol)

    d_exc_apprx = np.stack(
        np.sqrt(np.sum(omega_ab[i, :, :] * (r_ab ** 2)) / om[i]) for i in range(n_lowest))
    # binding energy approximated
    xs = np.stack((omega_ab[i, :, :] / r_ab) for i in range(n_lowest))
    xs[np.isinf(xs)] = 0
    binding_en_apprx = np.stack((np.sum(xs[i, :, :]) / om[i]) for i in range(n_lowest))

    descriptors = write_output_descriptors(
        d_exc, d_exc_apprx, d_he, sigma_h, sigma_e, r_eh, binding_en_apprx, n_lowest, omega, f)

    return descriptors

    # Find the index of the n_lowest excitations
#    idx = np.argsort(omega[:n_lowest])
    # Generate NTOs for each excited state
    # Solve the SVD for each state and store it into a list
#    xs = []
#    for i in range(n_lowest):
#        xs.append(np.linalg.svd(xia_I[:, :, idx[i] ]))
    # Do some renaming it for clearness
#    v_ip_I = np.array([xs[i][0] for i in range(n_lowest)])
#    lamb_p_I = np.array([xs[i][1] for i in range(n_lowest)])
#    w_ip_I = np.array([xs[i][2] for i in range(n_lowest)])
    # Compute the NTO participation ratio
#    pr_nto = np.stack(
#    ( np.sum(lamb_p_I[i, :] ** 2) ) ** 2 / np.sum(lamb_p_I[i, :] ** 4) for i in range(n_lowest) )


def write_output_descriptors(
        d_exc, d_exc_apprx, d_he, sigma_h, sigma_e, r_eh, binding_ex_apprx, n_lowest, omega, f):
    """
    ADD Documentation
    """
    au2ang = 0.529177249
    ex_output = np.empty((n_lowest, 10))
    ex_output[:, 0] = np.arange(n_lowest) + 1
    ex_output[:, 1] = d_exc * au2ang
    ex_output[:, 2] = d_exc_apprx * au2ang
    ex_output[:, 3] = d_he * au2ang
    ex_output[:, 4] = sigma_h * au2ang
    ex_output[:, 5] = sigma_e * au2ang
    ex_output[:, 6] = r_eh * au2ang
    ex_output[:, 7] = binding_ex_apprx * 27.211 / 2  # in eV
    ex_output[:, 8] = omega[:n_lowest] * 27.211
    ex_output[:, 9] = f[:n_lowest]

    return ex_output


def get_omega(d0I_ao, s, n_lowest):
    """
    ADD Documentation
    """
    return np.stack(
        np.trace(
            np.linalg.multi_dot([d0I_ao[i, :, :].T, s, d0I_ao[i, :, :], s]))
        for i in range(n_lowest))


def get_r_ab(mol):
    """
    ADD Documentation
    """
    coords = np.asarray([atom[1] for atom in mol])
    # Distance matrix between atoms A and B
    r_ab = cdist(coords, coords)
    return r_ab


def get_omega_ab(d0I_ao, s, n_lowest, mol, config):
    """
    ADD Documentation
    """
    # Lowdin transformation of the transition density matrix
    n_atoms = len(mol)
    s_sqrt = sqrtm(s)
    d0I_mo = np.stack(
        np.linalg.multi_dot([s_sqrt, d0I_ao[i, :, :], s_sqrt]) for i in range(n_lowest))

    # Compute the number of spherical functions for each atom
    n_sph_atoms = number_spherical_functions_per_atom(
        mol, config['package_name'], config['basis_name'], config['path_hdf5'])

    # Compute omega_ab
    omega_ab = np.zeros((n_lowest, n_atoms, n_atoms))
    for i in range(n_lowest):
        index_a = 0
        for a in range(n_atoms):
            index_b = 0
            for b in range(n_atoms):
                omega_ab[i, a, b] = np.sum(
                    d0I_mo[i, index_a:(index_a + n_sph_atoms[a]), index_b:(index_b + n_sph_atoms[b])] ** 2)
                index_b += n_sph_atoms[b]
            index_a += n_sph_atoms[a]

    return omega_ab


def get_exciton_positions(d0I_ao, s, moment, n_lowest, carrier):
    """
    ADD Documentation
    """
    def compute_component_hole(k):
        return np.stack(
            np.trace(
                np.linalg.multi_dot([d0I_ao[i, :, :].T, moment[k, :, :], d0I_ao[i, :, :], s]))
            for i in range(n_lowest))

    def compute_component_electron(k):
        return np.stack(
            np.trace(
                np.linalg.multi_dot([d0I_ao[i, :, :].T, s, d0I_ao[i, :, :],  moment[k, :, :]]))
            for i in range(n_lowest))

    def compute_component_he(k):
        return np.stack(
            np.trace(
                np.linalg.multi_dot(
                    [d0I_ao[i, :, :].T, moment[0, :, :], d0I_ao[i, :, :], moment[0, :, :]]))
            for i in range(n_lowest))

    if carrier == 'hole':
        return tuple(compute_component_hole(k) for k in range(3))
    elif carrier == 'electron':
        return tuple(compute_component_electron(k) for k in range(3))
    elif carrier == 'both':
        return tuple(compute_component_he(k) for k in range(3))
    else:
        raise RuntimeError("unkown option: {}".format(carrier))


def write_output_tddft(nocc, nvirt, omega, f, d_x, d_y, d_z, xia, e):
    """ Write out as a table in plane text"""

    excs = []
    for i in range(nocc):
        for a in range(nocc, nvirt + nocc):
            excs.append((i, a))

    output = np.empty((nocc*nvirt, 12))
    output[:, 0] = 0  # State number: we update it after reorder
    output[:, 1] = omega * h2ev  # State energy in eV
    output[:, 2] = f  # Oscillator strength
    output[:, 3] = d_x  # Transition dipole moment in the x direction
    output[:, 4] = d_y  # Transition dipole moment in the y direction
    output[:, 5] = d_z  # Transition dipole moment in the z direction
    # Weight of the most important excitation
    output[:, 6] = np.hstack(np.max(xia[:, i] ** 2) for i in range(nocc*nvirt))

    # Find the index of this transition
    index_weight = np.hstack(
        np.where(
            xia[:, i] ** 2 == np.max(
                xia[:, i] ** 2)) for i in range(nocc*nvirt)).reshape(nocc*nvirt)

    # Index of the hole for the most important excitation
    output[:, 7] = np.stack(excs[index_weight[i]][0] for i in range(nocc*nvirt)) + 1
    # These are the energies of the hole for the transition with the larger weight
    output[:, 8] = e[output[:, 7].astype(int) - 1] * h2ev
    # Index of the electron for the most important excitation
    output[:, 9] = np.stack(excs[index_weight[i]][1] for i in range(nocc*nvirt)) + 1
    # These are the energies of the electron for the transition with the larger weight
    output[:, 10] = e[output[:, 9].astype(int) - 1] * h2ev
    # This is the energy for the transition with the larger weight
    output[:, 11] = (e[output[:, 9].astype(int) - 1] - e[output[:, 7].astype(int) - 1]) * h2ev

    # Reorder the output in ascending order of energy
    output = output[output[:, 1].argsort()]
    # Give a state number in the correct order
    output[:, 0] = np.arange(nocc * nvirt) + 1

    return output


def number_spherical_functions_per_atom(mol, package_name, basis_name, path_hdf5):
    """
    ADD Documentation
    """
    with h5py.File(path_hdf5, 'r') as f5:
        xs = [f5['{}/basis/{}/{}/coefficients'.format(
            package_name, atom[0], basis_name)] for atom in mol]
        ys = [calc_orbital_Slabels(
            package_name, read_basis_format(
                package_name, path.attrs['basisFormat'])) for path in xs]

        return np.stack(sum(len(x) for x in ys[i]) for i in range(len(mol)))


def transition_density_charges(mol, config, s, c_ao):
    """
    ADD Documentation
    """
    n_atoms = len(mol)
    sqrt_s = sqrtm(s)
    c_mo = np.dot(sqrt_s, c_ao)
    # Size of the transition density tensor : n_atoms x n_mos x n_mos
    q = np.zeros((n_atoms, c_mo.shape[1], c_mo.shape[1]))
    n_sph_atoms = number_spherical_functions_per_atom(
        mol, config['package_name'], config['basis_name'], config['path_hdf5'])

    index = 0
    for i in range(n_atoms):
        q[i, :, :] = np.dot(
            c_mo[index:(index + n_sph_atoms[i]), :].T, c_mo[index:(index + n_sph_atoms[i]), :])
        index += n_sph_atoms[i]

    return q


def compute_MNOK_integrals(mol, xc_dft):
    """
    ADD Documentation
    """
    n_atoms = len(mol)
    r_ab = get_r_ab(mol)
    hardness_vec = np.stack(hardness(mol[i][0]) for i in range(n_atoms)).reshape(n_atoms, 1)
    hard = np.add(hardness_vec, hardness_vec.T)
    beta = xc(xc_dft)['beta1'] + xc(xc_dft)['ax'] * xc(xc_dft)['beta2']
    alpha = xc(xc_dft)['alpha1'] + xc(xc_dft)['ax'] * xc(xc_dft)['alpha2']
    if (xc(xc_dft)['type'] == 'pure'):
        gamma_J = np.zeros((n_atoms, n_atoms))
    else:
        gamma_J = np.power(
            1 / (np.power(r_ab, beta) + np.power((xc(xc_dft)['ax'] * hard), -beta)), 1/beta)
    gamma_K = np.power(1 / (np.power(r_ab, alpha) + np.power(hard, -alpha)), 1/alpha)

    return gamma_J, gamma_K


def construct_A_matrix_tddft(pqrs_J, pqrs_K, nocc, nvirt, xc_dft, e):
    """
    ADD Documentation
    """
    # This is the exchange integral entering the A matrix.
    #  It is in the format (nocc, nvirt, nocc, nvirt)
    k_iajb = pqrs_K[:nocc, nocc:, :nocc, nocc:].reshape(nocc*nvirt, nocc*nvirt)

    # This is the Coulomb integral entering in the A matrix.
    # It is in the format: (nocc, nocc, nvirt, nvirt)
    k_ijab_tmp = pqrs_J[:nocc, :nocc, nocc:, nocc:]

    # To get the correct order in the A matrix, i.e. (nocc, nvirt, nocc, nvirt),
    # we have to swap axes
    k_ijab = np.swapaxes(k_ijab_tmp, axis1=1, axis2=2).reshape(nocc*nvirt, nocc*nvirt)

    # They are in the m x m format where m is the number of excitations = nocc * nvirt
    a_mat = 2 * k_iajb - k_ijab

    # Generate a vector with all possible ea - ei energy differences
    e_diff = -np.subtract(
        e[:nocc].reshape(nocc, 1), e[nocc:].reshape(nvirt, 1).T).reshape(nocc*nvirt)
    np.fill_diagonal(a_mat, np.diag(a_mat) + e_diff)

    return a_mat

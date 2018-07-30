
from numba import jit, njit, float64
from numpy import (exp, log, pi, sqrt)
import numpy as np


@jit
def sab(gs1, gs2):
    """
    Primitive overlap terms calculated with the Obara-Saika recurrence relations,
    see: Molecular Electronic-Structure Theory. T. Helgaker, P. Jorgensen, J. Olsen.
    John Wiley & Sons. 2000, pages: 346-347.

    .. math::
        S_{i+1,j} = X_PA * S_{ij} + 1/(2*p) * (i * S_{i-1,j} + j * S_{i,j-1})
        S_{i,j+1} = X_PB * S_{ij} + 1/(2*p) * (i * S_{i-1,j} + j * S_{i,j-1})

   :param gs1: First Contracted Gaussian Function
   :param gs2: Second Contracted Gaussian Function
    """
    rab, rp, rpa, rpb, s00, prod = np.ones(6, dtype=np.float64)

    r1, l1, (c1, e1) = gs1
    r2, l2, (c2, e2) = gs2
    rab = distance(r1, r2)

    if neglect_integral(rab, e1, e2, 1e-10):
        return 0
    else:
        cte = sqrt(pi / (e1 + e2))
        u = e1 * e2 / (e1 + e2)
        p = 1.0 / (2.0 * (e1 + e2))
        for i in range(3):
            l1x = get_indexes(l1, i)
            l2x = get_indexes(l2, i)
            rp = (e1 * r1[i] + e2 * r2[i]) / (e1 + e2)
            rab = r1[i] - r2[i]
            rpa = rp - r1[i]
            rpb = rp - r2[i]
            s00 = cte * exp(-u * rab ** 2.0)
            # select the exponent of the multipole
            prod *= obaraSaikaMultipole(p, s00, rpa, rpb, 0, l1x, l2x, 0)

        return c1 * c2 * prod

@jit
def sab_efg(gs1, gs2, rc, e, f, g):
    """
    Primitive overlap terms calculated with the Obara-Saika recurrence relations,
    see: Molecular Electronic-Structure Theory. T. Helgaker, P. Jorgensen, J. Olsen.
    John Wiley & Sons. 2000, pages: 346-347.

    .. math::
        S^{e}_{i+1,j} = X_PA * S^{e}_{ij} + 1/(2*p) * (i * S{e}_{i-1,j} + j * S^{e}_{i,j-1} + e * S^{e-1}_{i,j})
        S^{e}_{i,j+1} = X_PB * S^{e}_{ij} + 1/(2*p) * (i * S{e}_{i-1,j} + j * S^{e}_{i,j-1} + e * S^{e-1}_{i,j}
        S^{e+1}_{i,j} = X_PC * S^{e}_{ij} + 1/(2*p) * (i * S{e}_{i-1,j} + j * S^{e}_{i,j-1} + e * S^{e-1}_{i,j})

    :param gs1: First Gaussian Primitive
    :param gs2: Second Gaussian Primitive
    """
    rab, rp, rpa, rpb, rpc, s00, prod = np.ones(7, dtype=np.float64)

    r1, l1, (c1, e1) = gs1
    r2, l2, (c2, e2) = gs2
    cte = sqrt(pi / (e1 + e2))
    u = e1 * e2 / (e1 + e2)
    p = 1.0 / (2.0 * (e1 + e2))
    multipoles = [e, f, g]

    i = 0 if e != 0 else (1 if f != 0 else 2)

    l1x = get_indexes(l1, i)
    l2x = get_indexes(l2, i)
    rp = (e1 * r1[i] + e2 * r2[i]) / (e1 + e2)
    rab = r1[i] - r2[i]
    rpa = rp - r1[i]
    rpb = rp - r2[i]
    rpc = rp - rc[i]
    s00 = cte * exp(-u * rab ** 2.0)
    # select the exponent of the multipole
    prod = obaraSaikaMultipole(p, s00, rpa, rpb, rpc, l1x, l2x, multipoles[i])

    return c1 * c2 * prod


@njit
def obaraSaikaMultipole(p, s00x,  xpa, xpb, xpc, i, j, e):
    """
    The  Obara-Saika Scheme to calculate overlap integrals. Explicit expressions
    for the s, p, and d orbitals for both de overlap and the integrals
    are written. Higher terms are calculated recursively.
    """
    if i < 0 or j < 0 or e < 0:
        return 0
    elif i == 0 and j == 0 and e == 0:
        return s00x
    elif i == 1 and j == 0 and e == 0:
        return xpa * s00x
    elif i == 0 and j == 1 and e == 0:
        return xpb * s00x
    elif i == 0 and j == 0 and e == 1:
        return xpc * s00x
    elif i == 1 and j == 1 and e == 0:
        return s00x * (xpa * xpb + p)
    elif i == 1 and j == 0 and e == 1:
        return s00x * (xpa * xpc + p)
    elif i == 0 and j == 1 and e == 1:
        return s00x * (xpb * xpc + p)
    elif i == 1 and j == 1 and e == 1:
        return s00x * (xpa * xpb * xpc + p * (xpa + xpb + xpc))
    elif i == 2 and j == 0 and e == 0:
        return s00x * (xpa ** 2 + p)
    elif i == 0 and j == 2 and e == 0:
        return s00x * (xpb ** 2 + p)
    elif i == 2 and j == 0 and e == 1:
        return s00x * ((xpa ** 2) * xpc + p * (2 * xpa + xpc))
    elif i == 0 and j == 2 and e == 1:
        return s00x * ((xpb ** 2) * xpc + p * (2 * xpb + xpc))
    elif i == 2 and j == 1 and e == 0:
        return s00x * ((xpa ** 2) * xpb + p * (2 * xpa + xpb))
    elif i == 1 and j == 2 and e == 0:
        return s00x * (xpa * (xpb ** 2) + p * (xpa + 2 * xpb))
    elif i == 2 and j == 1 and e == 1:
        return s00x * ((xpa ** 2) * xpb * xpc + p *
                       ((xpa ** 2) + 2 * xpa * xpb + 2 * xpa * xpc +
                        xpb * xpc + 3 * p))
    elif i == 1 and j == 2 and e == 1:
        return s00x * (xpa * (xpb ** 2) * xpc + p *
                       ((xpb ** 2) + 2 * xpa * xpb + 2 * xpb * xpc +
                        xpa * xpc + 3 * p))

    # From higher order spin numbers only recursive relations are used.
    elif i >= 1:
        return xpa * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 1, j, e) + \
            p * ((i - 1) * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 2, j, e) +
                 j * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 1, j - 1, e) +
                 e * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 1, j, e - 1))

    elif j >= 1:
        return xpb * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j - 1, e) + \
            p * (i * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 1, j - 1, e) +
                 (j - 1) * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j - 2, e) +
                 e * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j - 1, e - 1))

    elif e >= 1:
        return xpc * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j, e - 1) + \
            p * (i * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 1, j, e - 1) +
                 j * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j - 1, e - 1) +
                 (e - 1) * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j, e - 2))


@njit
def neglect_integral(r, e1, e2, accuracy):
    """
    Compute whether an overlap integral should be neglected
    """
    ln = -log(accuracy) * ((1 / e1) + (1 / e2))

    return (r ** 2) > ln


@njit
def distance(xs, ys):
    """
    Distance between 2 points
    """
    acc = 0
    for x, y in zip(xs, ys):
        acc += (x - y) ** 2

    return sqrt(acc)


dict_indexes = [
    ("S",    0, 0, 0),
    ("Px",   1, 0, 0),
    ("Py",   0, 1, 0),
    ("Pz",   0, 0, 1),
    ("Dxx",  2, 0, 0),
    ("Dxy",  1, 1, 0),
    ("Dxz",  1, 0, 1),
    ("Dyy",  0, 2, 0),
    ("Dyz",  0, 1, 1),
    ("Dzz",  0, 0, 2),
    ("Fxxx", 3, 0, 0),
    ("Fxxy", 2, 1, 0),
    ("Fxxz", 2, 0, 1),
    ("Fxyy", 1, 2, 0),
    ("Fxyz", 1, 1, 1),
    ("Fxzz", 1, 0, 2),
    ("Fyyy", 0, 3, 0),
    ("Fyyz", 0, 2, 1),
    ("Fyzz", 0, 1, 2),
    ("Fzzz", 0, 0, 3)
]


# Create a numpy array using the Indexes
indexes = np.array(dict_indexes, dtype=[('lorb', 'S4'), ('x', '>i4'), ('y', '>i4'), ('z', '>i4')])


@jit
def get_indexes(key, index):
    """ Replace the orbital index dictionary with a numpy record """
    k = key.encode()
    tup = indexes[np.where(indexes['lorb'] == k)]
    return tup[0][index + 1]

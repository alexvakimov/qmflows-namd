
from numba import jit, njit, float64
from numpy import (exp, log, pi, sqrt)
import numpy as np


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

@njit
def test_numba():
    x = get_indexes('Fxxz', 0)
    print(x)

if __name__ == "__main__":
    test_numba()

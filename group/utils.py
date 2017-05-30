"""Helper functions used in the package"""

import os
import unittest
import numpy as np
import scipy.misc

# Rounds real and imaginary part of a complex number to zero if smaller than prec
def clean_complex(data, prec=1e-6):
    _data = np.asarray(data)
    for x in np.nditer(_data, op_flags=["readwrite"]):
        tmp = x.copy()
        if np.abs(x) < prec:
            tmp = complex(0.,0.)
        else:
            if np.abs(x.real) < prec:
                tmp = complex(0., x.imag)
            if np.abs(x.imag) < prec:
                tmp = complex(x.real, 0.)
        # reset and set the value of x
        x[...] *= 0.
        x[...] += tmp
    return _data

# Rounds real number to zero if smaller than prec
def clean_real(data, prec=1e-6):
    _data = np.asarray(data)
    for x in np.nditer(_data, op_flags=["readwrite"]):
        if np.abs(x) < prec:
            x[...] *= 0.
    return _data

# Compare two np.ndarrays 
def _eq(data1, data2=None, prec=1e-6):
    if data2 is None:
        return np.all(np.abs(data1) < prec)
    else:
        return np.all(np.abs(data1-data2) < prec)

def check_array(d1, d2, msg=None):
    c = np.isclose(d1,d2)
    if not np.all(c):
        if msg is not None:
            raise unittest.TestCase.failureException(msg)
        else:
            string = ["Arrays are not close elementwise.\n\n",
                "array 1:\n%r\n\narray 2:\n%r" % (d1,d2)]
            raise unittest.TestCase.failureException("".join(string))

def binomial(n, k):
    return scipy.misc.comb(n, k)

def gram_schmidt(v1, v2, prec=1e-6):
    """returns the part of v1 perpendicular to v2"""
    _v1 = np.asarray(v1)
    _v2 = np.asarray(v2)
    n1 = np.vdot(_v2, _v1)
    if np.abs(n1) < prec:
        n = np.sqrt(np.vdot(_v1, _v1))
        if np.abs(n) > prec:
            _v1 /= n
        return _v1
    n2 = np.sqrt(np.vdot(_v2, _v2))
    res = _v1 - n1/n2*_v2
    n = np.sqrt(np.vdot(res, res))
    if np.abs(n) > prec:
        res /= n
    return res

def ensure_write(filename, verbose=False):
    _d = os.path.dirname(filename)
    _d = os.path.normpath(_d)
    if not os.path.exists(_d):
        os.makedirs(_d)
        if verbose:
            print("created path %s" % _d)
    if verbose and os.path.isfile(filename):
        print("file exists and will be overwritten:\n%s" % filename)


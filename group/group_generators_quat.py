"""Generators for group representatives, based on quaternions."""

import numpy as np

import utils
import quat

# Pauli matrices 2d representation
_pmatrix = np.asarray(
        [[[1., 0.], [0., 1.]],
         [[0., 1.], [1., 0.]],
         [[0., -1.j], [1.j, 0.]],
         [[1., 0.], [0., -1.]]], dtype=complex)


# Generate matrix representation of elements for spin 0 (1d)
# Delegates to Q_new.R_matrix()
def genJ0(elements, inv=False, U=None):
    res = np.zeros((len(elements), 1, 1), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(0) * el.i
    else:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(0)
    return res

# Generate matrix representation of elements for spin 1/2 (2d)
# Delegates to Q_new.R_matrix()
def genJ1_2(elements, inv=False, U=None):
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(0.5) * el.i
    else:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(0.5)
    return res

# Generate matrix representation of elements for spin 1 (3d)
# Delegates to Q_new.R_matrix()
def genJ1(elements, inv=False, U=None):
    res = np.zeros((len(elements), 3, 3), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(1) * el.i
    else:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(1)
    return res

# Generate matrix representation of elements for spin 3/2 (4d)
# Delegates to Q_new.R_matrix()
def genJ3_2(elements, inv=False, U=None):
    res = np.zeros((len(elements), 4, 4), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(1.5) * el.i
    else:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(1.5)
    return res

# Generate A1 representation of elements for 1d
# All elements are +-1 depending on inversion of the group
def gen1D(elements, inv=False, U=None):
    if not inv:
        return np.ones((len(elements), 1, 1), dtype=complex)
    else:
        res = np.ones((len(elements), 1, 1), dtype=complex)
        for i, el in enumerate(elements):
            res[i] *= el.i
        return res

# Generate A1 representation of elements for 2d
# Results analytically calculated from quaternions
# TODO: How?
def gen2D(elements, inv=False, U=None):
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            tmp = np.asarray(
                    [[complex(el.q[0], -el.q[3]), complex(-el.q[2], -el.q[1])],
                     [complex(el.q[2], -el.q[1]), complex( el.q[0],  el.q[3])]],
                    dtype=complex)
            res[i] = tmp.copy() * el.i
    else:
        for i, el in enumerate(elements):
            tmp = np.asarray(
                    [[complex(el.q[0], -el.q[3]), complex(-el.q[2], -el.q[1])],
                     [complex(el.q[2], -el.q[1]), complex( el.q[0],  el.q[3])]],
                    dtype=complex)
            res[i] = tmp.copy()
    return res

def gen3D(elements, inv=False, U=None):
    res = np.zeros((len(elements), 3, 3), dtype=complex)
    for i, el in enumerate(elements):
        res[i] = el.rotation_matrix(inv)
    return res

def gen4D(elements, inv=False, U=None):
    res = np.zeros((len(elements), 4, 4), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            tmp = np.asarray(
                    [[el.q[0], -el.q[1], -el.q[2], -el.q[3]],
                     [el.q[1],  el.q[0], -el.q[3],  el.q[2]],
                     [el.q[2],  el.q[3],  el.q[0], -el.q[1]],
                     [el.q[3], -el.q[2],  el.q[1],  el.q[0]]], dtype=complex)
            res[i] = tmp.copy() * el.i
    else:
        for i, el in enumerate(elements):
            tmp = np.asarray(
                    [[el.q[0], -el.q[1], -el.q[2], -el.q[3]],
                     [el.q[1],  el.q[0], -el.q[3],  el.q[2]],
                     [el.q[2],  el.q[3],  el.q[0], -el.q[1]],
                     [el.q[3], -el.q[2],  el.q[1],  el.q[0]]], dtype=complex)
            res[i] = tmp.copy()
    return res

# special cases
def compare_quat(elem, sign=False):
    if sign:
        for i, q in enumerate(quat.qPar):
            if elem.comp(q):
                return i
            elif elem.comp(np.negative(q)):
                return -i
    else:
        for i, q in enumerate(quat.qPar):
            if elem.comp(q) or elem.comp(np.negative(q)):
                return i
    raise RuntimeError("element not identified: %r" % (elem.str()))

################################################################################
# Generate representations from analytic solution

def genEpCMF(elements, inv=False, U=None):
    if U is None:
        U = np.identity(2)
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    I  = complex( 1.0, 0.0 )
    O  = complex( 0.0, 0.0 )
    E  = np.exp(2.j*np.pi/3.)
    F = E.conj()
    m1 = np.asarray([[ I, O ],
                     [ O, I ]], dtype=complex)
    m2 = np.asarray([[ E, O ],
                     [ O, F ]], dtype=complex)
    m3 = np.asarray([[ F, O ],
                     [ O, E ]], dtype=complex)
    m4 = np.asarray([[ O, F ],
                     [ E, O ]], dtype=complex)
    m5 = np.asarray([[ O, E ],
                     [ F, O ]], dtype=complex)
    m6 = np.asarray([[ O, I ],
                     [ I, O ]], dtype=complex)
    m1list = [x+y*24 for x in [ 0, 1, 2, 3] for y in range(4)]
    m2list = [x+y*24 for x in [ 4, 5, 6, 7] for y in range(4)]
    m3list = [x+y*24 for x in [ 8, 9,10,11] for y in range(4)]
    m4list = [x+y*24 for x in [12,15,21,23] for y in range(4)]
    m5list = [x+y*24 for x in [13,16,20,22] for y in range(4)]
    m6list = [x+y*24 for x in [14,17,18,19] for y in range(4)]
    for i, elem in enumerate(elements):
        r = compare_quat(elem)
        if r in m1list:
            res[i] = m1
        elif r in m2list:
            res[i] = m2
        elif r in m3list:
            res[i] = m3
        elif r in m4list:
            res[i] = m4
        elif r in m5list:
            res[i] = m5
        elif r in m6list:
            res[i] = m6
        else:
            raise RuntimeError("element not identified")
        #res[i] = U.dot(res[i].dot(U))
        res[i] = U.dot(res[i].dot(U.conj().T))
    return res

def genEpCMF_old(elements, inv=False, U=None):
    if U is None:
        U = np.identity(2)
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    E  = complex( 0.5*np.sqrt(3.0), 0.0 )
    H  = complex( 0.5, 0.0 )
    O  = complex( 0.0, 0.0 )
    I  = complex( 1.0, 0.0 )
    m1 = np.asarray([[ I, O ],
                     [ O, I ]], dtype=complex)
    m2 = np.asarray([[-H, E ],
                     [-E,-H ]], dtype=complex)
    m3 = np.asarray([[-H,-E ],
                     [ E,-H ]], dtype=complex)
    m4 = np.asarray([[ I, O ],
                     [ O,-I ]], dtype=complex)
    m5 = np.asarray([[-H,-E ],
                     [-E, H ]], dtype=complex)
    m6 = np.asarray([[-H, E ],
                     [ E, H ]], dtype=complex)
    m1list = [x+y*24 for x in [ 0, 1, 2, 3] for y in range(4)]
    m2list = [x+y*24 for x in [ 4, 5, 6, 7] for y in range(4)]
    m3list = [x+y*24 for x in [ 8, 9,10,11] for y in range(4)]
    m4list = [x+y*24 for x in [14,17,18,19] for y in range(4)]
    m5list = [x+y*24 for x in [13,16,20,22] for y in range(4)]
    m6list = [x+y*24 for x in [12,15,21,23] for y in range(4)]
    for i, elem in enumerate(elements):
        r = compare_quat(elem)
        if r in m1list:
            res[i] = m1
        elif r in m2list:
            res[i] = m2
        elif r in m3list:
            res[i] = m3
        elif r in m4list:
            res[i] = m4
        elif r in m5list:
            res[i] = m5
        elif r in m6list:
            res[i] = m6
        else:
            raise RuntimeError("element not identified")
        res[i] = U.dot(res[i].dot(U.conj().T))
    return res

def genEpMF1(elements, inv=False, U=None):
    if U is None:
        U = np.identity(2)
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    O  = complex(  0.0, 0.0 )
    I  = complex(  1.0, 0.0 )
    J  = complex(  0.0, 1.0 )
    m1 = np.asarray([[ I, O],
                     [ O, I]], dtype=complex)
    m2 = np.asarray([[-J, O],
                     [ O, J]], dtype=complex)
    m3 = np.asarray([[ J, O],
                     [ O,-J]], dtype=complex)
    m4 = np.asarray([[-I, O],
                     [ O,-I]], dtype=complex)
    m5 = np.asarray([[ O,-I],
                     [-I, O]], dtype=complex)
    m6 = np.asarray([[ O, I],
                     [ I, O]], dtype=complex)
    m7 = np.asarray([[ O, J],
                     [-J, O]], dtype=complex)
    m8 = np.asarray([[ O,-J],
                     [ J, O]], dtype=complex)
    # hard-coded because I don't know which elements
    # of qPar contribute
    m1list = [x+y*24 for x in [ 0] for y in range(4)]
    m2list = [x+y*24 for x in [14] for y in range(4)]
    m3list = [x+y*24 for x in [17] for y in range(4)]
    m4list = [x+y*24 for x in [ 3] for y in range(4)]
    m5list = [x+y*24 for x in [ 1] for y in range(4)]
    m6list = [x+y*24 for x in [ 2] for y in range(4)]
    m7list = [x+y*24 for x in [18] for y in range(4)]
    m8list = [x+y*24 for x in [19] for y in range(4)]
    for i, elem in enumerate(elements):
        r = compare_quat(elem)
        if r in m1list:
            res[i] = m1
        elif r in m2list:
            res[i] = m2
        elif r in m3list:
            res[i] = m3
        elif r in m4list:
            res[i] = m4
        elif r in m5list:
            res[i] = m5
        elif r in m6list:
            res[i] = m6
        elif r in m7list:
            res[i] = m7
        elif r in m8list:
            res[i] = m8
        else:
            print("what to do with %d" % r)
        #res[i] = U.dot(res[i].dot(U))
        res[i] = U.dot(res[i].dot(U.conj().T))
    return res

def genEpMF1_old(elements, inv=False):
    if U is None:
        U = np.identity(2)
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    O  = complex(  0.0, 0.0 )
    I  = complex(  1.0, 0.0 )
    J  = complex(  0.0, 1.0 )
    m1 = np.asarray([[ I, O],
                     [ O, I]], dtype=complex)
    m2 = np.asarray([[-I, O],
                     [ O,-I]], dtype=complex)
    m3 = np.asarray([[ O, I],
                     [ I, O]], dtype=complex)
    m4 = np.asarray([[ O,-I],
                     [-I, O]], dtype=complex)
    m5 = np.asarray([[ J, O],
                     [ O,-J]], dtype=complex)
    m6 = np.asarray([[-J, O],
                     [ O, J]], dtype=complex)
    m7 = np.asarray([[ O,-J],
                     [ J, O]], dtype=complex)
    m8 = np.asarray([[ O, J],
                     [-J, O]], dtype=complex)
    # hard-coded because I don't know which elements
    # of qPar contribute
    m1list = [x+y*8 for x in [0] for y in range(2)]
    m2list = [x+y*8 for x in [1] for y in range(2)]
    m3list = [x+y*8 for x in [4] for y in range(2)]
    m4list = [x+y*8 for x in [5] for y in range(2)]
    m5list = [x+y*8 for x in [2] for y in range(2)]
    m6list = [x+y*8 for x in [3] for y in range(2)]
    m7list = [x+y*8 for x in [7] for y in range(2)]
    m8list = [x+y*8 for x in [6] for y in range(2)]
    for i, elem in enumerate(elements):
        if i in m1list:
            res[i] = m1
        elif i in m2list:
            res[i] = m2
        elif i in m3list:
            res[i] = m3
        elif i in m4list:
            res[i] = m4
        elif i in m5list:
            res[i] = m5
        elif i in m6list:
            res[i] = m6
        elif i in m7list:
            res[i] = m7
        elif i in m8list:
            res[i] = m8
        else:
            print("what to do with %d" % i)
        #res[i] = U.dot(res[i].dot(U))
        res[i] = U.dot(res[i].dot(U.conj().T))
    return res

def genT1CMF(elements, inv=False, U=None):
    if U is None:
        U = np.identity(3)
    s = 1./np.sqrt(2)
    res = np.zeros((len(elements), 3, 3), dtype=complex)
    pars = np.zeros((9,), dtype=complex)
    for i, elem in enumerate(elements):
        pars.fill(0.)
        r = compare_quat(elem)
        if r in [0,2,13,16,]:
            pars[0] = 1.
        elif r in [1,3,20,22]:
            pars[0] = -1.
        if r in [8,10,15,21]:
            pars[1] = 1.
        elif r in [9,11,12,23]:
            pars[1] = -1.
        if r in [6,7,17,19]:
            pars[2] = 1.
        elif r in [4,5,14,18]:
            pars[2] = -1.
        if r in [5,7,15,23]:
            pars[3] = 1.
        elif r in [4,6,12,21]:
            pars[3] = -1.
        if r in [0,3,14,17]:
            pars[4] = 1.
        elif r in [1,2,18,19]:
            pars[4] = -1.
        if r in [9,10,13,22]:
            pars[5] = 1.
        elif r in [8,11,16,20]:
            pars[5] = -1.
        if r in [8,9,17,18]:
            pars[6] = 1.
        elif r in [10,11,14,19]:
            pars[6] = -1.
        if r in [5,6,16,22]:
            pars[7] = 1.
        elif r in [4,7,13,20]:
            pars[7] = -1.
        if r in [0,1,12,15]:
            pars[8] = 1.
        elif r in [2,3,21,23]:
            pars[8] = -1.
        res[i][0,0] = pars[0]*1.
        res[i][0,1] = pars[1]*1.j
        res[i][0,2] = pars[2]*1.j
        res[i][1,0] = pars[3]*1.j
        res[i][1,1] = pars[4]*1.
        res[i][1,2] = pars[5]*1.
        res[i][2,0] = pars[6]*1.j
        res[i][2,1] = pars[7]*1.
        res[i][2,2] = pars[8]*1.
        #if i == 4:
        #    print("debugging element 4")
        #    print(res[i])
        #    print("R*U^\dagger")
        #    tmp = res[i].dot(U.conj().T)
        #    print(tmp)
        #    print("U*R*U^\dagger")
        #    tmp1 = U.dot(tmp)
        #    print(tmp1)
        #    res[i] = tmp1
        #    print
        #else:
        #    res[i] = U.dot(res[i].dot(U))
        res[i] = U.dot(res[i].dot(U.conj().T))
        if inv:
            res[i] *= elem.i
    return res

def genF1CMF(elements, inv=False, U=None):
    if U is None:
        U = np.identity(4)
    res = np.zeros((len(elements), 4, 4), dtype=complex)
    d = np.exp(2.j*np.pi/24)
    md = -d
    dc = d.conj()
    mdc = -d.conj()
    _id = 1.j*d
    mid = -1.j*d
    idc = 1.j*d.conj()
    midc = -1.j*d.conj()
    e = np.exp(2.j*np.pi/ 8)
    me = -e
    ec = e.conj()
    mec = -e.conj()
    n = np.exp(2.j*np.pi/ 3)
    mn = -n
    nc = n.conj()
    mnc = -n.conj()
    _in = 1.j*n
    _min = -1.j*n
    inc = 1.j*n.conj()
    minc = -1.j*n.conj()
    k = -1.j
    _mat = np.asarray(
            [[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],
             [[0., k,0.,0.],[ k,0.,0.,0.],[0.,0.,0., k],[0.,0., k,0.]],
             [[0.,-1,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,-1],[0.,0.,1.,0.]],
             [[ k,0.,0.,0.],[0.,1j,0.,0.],[0.,0., k,0.],[0.,0.,0.,1j]],
             [[idc,dc,0.,0.],[idc,mdc,0.,0.],[0.,0.,md,_id],[0.,0.,md,mid]],
             [[idc,mdc,0.,0.],[midc,mdc,0.,0.],[0.,0.,md,mid],[0.,0.,d,mid]],
             [[mdc,idc,0.,0.],[dc,idc,0.,0.],[0.,0.,mid,md],[0.,0.,_id,md]],
             [[mdc,midc,0.,0.],[mdc,idc,0.,0.],[0.,0.,mid,d],[0.,0.,mid,md]],
             [[mid,mid,0.,0.],[d,md,0.,0.],[0.,0.,mdc,mdc],[0.,0.,midc,idc]],
             [[mid,_id,0.,0.],[md,md,0.,0.],[0.,0.,mdc,dc],[0.,0.,idc,idc]],
             [[md,d,0.,0.],[mid,mid,0.,0.],[0.,0.,idc,midc],[0.,0.,mdc,mdc]],
             [[md,md,0.,0.],[_id,mid,0.,0.],[0.,0.,idc,idc],[0.,0.,dc,mdc]],
             [[0.,0.,nc,minc],[0.,0.,minc,nc],[n,_min,0.,0.],[_min,n,0.,0.]],
             [[0.,0.,n,mn],[0.,0.,n,n],[nc,mnc,0.,0.],[nc,nc,0.,0.]],
             [[0.,0.,ec,0.],[0.,0.,0.,e],[ec,0.,0.,0.],[0.,e,0.,0.]],
             [[0.,0.,nc,inc],[0.,0.,inc,nc],[n,_in,0.,0.],[_in,n,0.,0.]],
             [[0.,0.,n,n],[0.,0.,mn,n],[nc,nc,0.,0.],[mnc,nc,0.,0.]],
             [[0.,0.,e,0.],[0.,0.,0.,ec],[e,0.,0.,0.],[0.,ec,0.,0.]],
             [[0.,0.,0.,me],[0.,0.,ec,0.],[0.,me,0.,0.],[ec,0.,0.,0.]],
             [[0.,0.,0.,mec],[0.,0.,e,0.],[0.,mec,0.,0.],[e,0.,0.,0.]],
             [[0.,0.,_min,_min],[0.,0.,_min,_in],[minc,minc,0.,0.],[minc,inc,0.,0.]],
             [[0.,0.,inc,nc],[0.,0.,mnc,minc],[_in,n,0.,0.],[mn,_min,0.,0.]],
             [[0.,0.,_in,_min],[0.,0.,_min,_min],[inc,minc,0.,0.],[minc,minc,0.,0.]],
             [[0.,0.,minc,nc],[0.,0.,mnc,inc],[_min,n,0.,0.],[mn,_in,0.,0.]]
             ], dtype=complex)
    _check = np.asarray([-1.,0.,0.,0.])
    _r = range(4,14) + [15,16] + range(20,24)
    for i, elem in enumerate(elements):
        r = compare_quat(elem, sign=True)
        if r < 0 or (r == 0 and np.allclose(elem.q,_check)):
            res[i] = -_mat[-r]
        else:
            res[i] = _mat[r]

        if np.abs(r) in _r:
            res[i] /= np.sqrt(2)
        res[i] = U.dot(res[i].dot(U.conj().T))
        if inv:
            res[i] *= elem.i
    return res

if __name__ == "__main__":
    print("for checks execute the test script")


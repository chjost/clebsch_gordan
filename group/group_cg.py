"""Class for the clebsch-gordan coefficients of a group."""

import numpy as np
import itertools as it
import pandas as pd
from pandas import Series, DataFrame

import group_class
import utils
from rotations import _all_rotations

class OhCG(object):
    def __init__(self, p, p1, p2, groups=None):
        """p, p1, and p2 are the magnitudes of the momenta.
        """
        self.prec = 1e-6
        # save the norm of the momenta for the combined system
        # and each particle
        self.p = p
        self.p1 = p1
        self.p2 = p2
        # lookup table for reference momenta
        lpref = [np.asarray([0.,0.,0.]), np.asarray([0.,0.,1.]), 
                 np.asarray([1.,1.,0.]), np.asarray([1.,1.,1.])]
        # save reference momenta
        self.pref = lpref[p]
        self.pref1 = lpref[p1]
        self.pref2 = lpref[p2]

        # get the basic groups
        if groups is None:
            self.g0 = None
            self.g = None
            self.g1 = None
            self.g2 = None
        else:
            self.g0 = groups[0]
            self.g = groups[p]
            self.g1 = groups[p1]
            self.g2 = groups[p2]

        # get the cosets, always in the maximal group (2O here)
        # is set to None if at least one group is None
        self.coset1 = self.gen_coset(self.g1)
        self.coset2 = self.gen_coset(self.g2)
        #print(self.coset1)
        #print(self.coset2)

        # generate the allowed momentum combinations and sort them into cosets
        self.gen_momenta()
        if groups is not None:
            self.sort_momenta()

        # calculate induced rep gamma
        # here for p1 and p2 the A1(A2) irreps are hard-coded
        # since only these contribute to pi-pi scattering
        if groups is None:
            self.gamma1 = None
            self.gamma2 = None
        else:
            irstr = "A1" if p1 < 1e-6 else "A2"
            self.gamma1 = self.gen_ind_reps(self.g, self.g1, irstr, self.coset1)
            irstr = "A1" if p2 < 1e-6 else "A2"
            self.gamma2 = self.gen_ind_reps(self.g, self.g2, irstr, self.coset2)
        #print(self.gamma1[:5])

        self.irreps = []
        self.cgs = []

        # choose mu1, mu2 and i1, i2, according to Dudek paper
        # since A1 and A2 are 1D, set mu to 0
        self.mu1 = 0
        self.mu2 = 0
        if self.p == 0:
            # in this case chose the hightest option
            self.i1 = -1
            self.i2 = -1
            self.i1i2 = [(self.pref, -1, -1)]
        else:
            self.i1i2 = []
            for m in self.momenta:
                for ((m1, i1), (m2, i2)) in it.product(self.smomenta1, self.smomenta2):
                    if utils._eq(m1+m2-m):
                        self.i1i2.append((m,i1,i2))
                        self.i1 = i1
                        self.i2 = i2
                        break

    def gen_momenta(self):
        pm = 4 # maximum component in each direction
        def _abs(x):
            return np.dot(x, x) <= pm
        def _abs1(x,a):
            return np.dot(x, x) == a
        gen = it.ifilter(_abs, it.product(range(-pm,pm+1), repeat=3))
        lp3 = [np.asarray(y, dtype=int) for y in gen]
        self.momenta = [y for y in it.ifilter(lambda x: _abs1(x,self.p), lp3)]
        self.momenta1 = [y for y in it.ifilter(lambda x: _abs1(x,self.p1), lp3)]
        self.momenta2 = [y for y in it.ifilter(lambda x: _abs1(x,self.p2), lp3)]
        
        self.allmomenta = []
        # only save allowed momenta combinations
        for p in self.momenta:
            for p1 in self.momenta1:
                for p2 in self.momenta2:
                    if utils._eq(p1+p2-p):
                        self.allmomenta.append((p, p1, p2))

    def gen_coset(self, g1):
        """Cosets contain the numbers of the rotation objects
        """
        if self.g0 is None or g1 is None:
            return None
        n = int(self.g0.order/g1.order)
        if n == 0:
            raise RuntimeError("number of cosets is 0!")
        coset = np.zeros((n, g1.order), dtype=int)
        l = self.g0.order
        l1 = g1.order
        # set the subgroup
        count = 0
        for r in range(l1):
            elem = g1.lrotations[r]
            if elem in self.g0.lrotations:
                coset[0, count] = elem
                count += 1
        # calc the cosets
        uniq = np.unique(coset)
        cnum = 1 # coset number
        for elem in self.g0.lrotations:
            if elem in uniq:
                continue
            count = 0
            for elem1 in g1.lrotations:
                if elem1 in self.g0.lrotations:
                    # multiplication table contains numbers
                    # [0, g.order), so lookup the element
                    look = self.g0.lrotations.index(elem1)
                    el = self.g0.tmult[look, elem]
                    coset[cnum, count] = self.g0.lrotations[el]
                    count += 1
            cnum += 1
            uniq = np.unique(coset)
        if len(uniq) != self.g0.order:
            print("some elements got lost!")
        if cnum != n:
            print("some coset not found!")
        return coset

    def gen_ind_reps(self, g, g1, irstr, coset):
        ir = g1.instances[g1.lirreps.index(irstr)]
        dim = ir.dim
        ndim = (self.g0.order, coset.shape[0]*dim, coset.shape[0]*dim)
        gamma = np.zeros(ndim, dtype=complex)
        for ind, r in enumerate(self.g0.lrotations):
            # take the first elements of the coset as representatives
            for c1, rj in enumerate(coset[:,0]):
                # translate to multiplication table
                el1 = self.g0.lrotations.index(rj)
                # get element
                rrj = self.g0.tmult[ind,el1]
                ind1 = slice(c1*dim, (c1+1)*dim)
                for c2, ri in enumerate(coset[:,0]):
                    # translate to multiplication table and get inverse
                    el2 = self.g0.lrotations.index(ri)
                    riinv = self.g0.Inverse(el2)
                    # get element
                    riinvrrj = self.g0.tmult[riinv, rrj]
                    # translate to rotation element and check
                    # if in subgroup
                    el3 = self.g0.lrotations[riinvrrj]
                    if el3 not in coset[0]:
                        continue
                    # if in subgroup, look up position of element
                    elem = g1.lrotations.index(el3)
                    ind2 = slice(c2*dim,(c2+1)*dim)
                    # set induced representation
                    gamma[ind, ind1, ind2] = ir.mx[elem]
        return gamma

    def sort_momenta(self):
        # check if cosets exists
        if self.coset1 is None or self.coset2 is None:
            self.smomenta1 = None
            self.smomenta2 = None
            return
        # search for conjugacy class so that
        # R*p_ref = p
        res1 = []
        res2 = []
        for p, p1, p2 in self.allmomenta:
            #print("momentum coset search")
            done = False
            # check if already in list
            for r in res1:
                if utils._eq(r[0],p1):
                    done=True
                    break
            # if not get coset
            if not done:
                #print("%r not in list" % p1)
                for i, c in enumerate(self.coset1):
                    t = self.check_coset(self.pref1, p1, c)
                    #print(t)
                    if np.all(t):
                        #print(" in coset %d" %i)
                        res1.append((p1, i))
                        break
                    #else:
                    #    print(" not in coset %d" %i)
            #else:
            #    print("%r already in list" % p1)
            done = False
            # check if already in list
            for r in res2:
                if utils._eq(r[0],p2):
                    done=True
                    break
            if not done:
                for i, c in enumerate(self.coset2):
                    t = self.check_coset(self.pref2, p2, c)
                    if np.all(t):
                        res2.append((p2, i))
                        break
        self.smomenta1 = res1
        if len(self.smomenta1) != len(self.momenta1):
            print("some vectors not sorted")
        self.smomenta2 = res2
        if len(self.smomenta2) != len(self.momenta2):
            print("some vectors not sorted")

    def check_coset(self, pref, p, coset):
        res = []
        for elem in coset:
            rot = _all_rotations[elem]
            rvec = rot.rot_vector(pref)
            c1 = utils._eq(rvec, p)
            c2 = utils._eq(rvec, -p)
            if c1 or c2:
                res.append(True)
            else:
                res.append(False)
        return res

    def check_all_cosets(self, p, p1, p2):
        j1, j2 = None, None
        i1, i2 = None, None
        for m, j in self.smomenta1:
            if utils._eq(p1,m):
                j1 = j
                break
        for m, j in self.smomenta2:
            if utils._eq(p2,m):
                j2 = j
                break
        for m, k1, k2 in self.i1i2:
            if utils._eq(p, m):
                i1, i2 = k1, k2
                break
        return j1, j2, i1, i2

    def calc_pion_cg(self, p, p1, p2, irname):
        """Calculate the elements of the Clebsch-Gordan matrix.

        Assumes that p=p1+p2, where all three are 3-vectors.
        """
        # get irrep of group g
        ir = self.g.instances[self.g.lirreps.index(irname)]
        # j1 and j2 are the conjugacy classes containing
        # the given momenta p1 and p2
        j1, j2, i1, i2 = self.check_all_cosets(p, p1, p2)
        #print(j1, j2, p1)
        cg = np.zeros((ir.dim,ir.dim), dtype=complex)
        for ind, r in enumerate(self.g.lrotations):
            rep = ir.mx[ind]
            # hard coded for pi-pi scattering
            g1 = self.gamma1[r,j1, i1]
            if utils._eq(g1):
                continue
            g2 = self.gamma2[r,j2, i2]
            if utils._eq(g2):
                continue
            cg += rep.conj()*g1*g2
        cg *= float(ir.dim)/self.g.order
        return cg
    
    def get_pion_cg(self, irname):
        try:
            ind = self.irreps.index(irname)
            return irname, self.cgs[ind], self.allmomenta
        except:
            pass
        result = []
        # iterate over momenta
        for p, p1, p2 in self.allmomenta:
            res = self.calc_pion_cg(p, p1, p2, irname)
            if res is None:
                continue
            result.append(res)
        result = np.asarray(result)
        # check if all coefficients are zero
        if utils._eq(result):
            cgs = None
        else:
            # orthonormalize the basis
            cgs = self._norm_cgs(result)
        self.irreps.append(irname)
        self.cgs.append(cgs)
        return irname, cgs, self.allmomenta

    def _norm_cgs(self, data):
        # prepare result array
        res = np.zeros(data.shape[:-1], dtype=complex)
        # sort by final momentum, so that all final momenta are
        # normalized seperately
        ind = [[] for x in self.momenta]
        for i, m in enumerate(self.allmomenta):
            for j, fm in enumerate(self.momenta):
                if np.array_equal(m[0], fm):
                    ind[j].append(i)
                    break
        # norm the data
        # set starting variables
        mup = 0
        for i in range(data.shape[1]):
            for j in ind:
                tmp = data[j,i,mup]
                if np.any(tmp):
                    norm = np.sqrt(np.vdot(tmp, tmp))
                    res[j,i] = tmp/norm
            mup += 1
        return res

def display(data, mom, empty=None):
    def _d1(data):
        tmp = ["%2d" % x for x in data]
        tmp = ",".join(tmp)
        tmp = "".join(("(", tmp, ")"))
        return tmp
    def _d2(data):
        tmp = ["%+.3f%+.3fj" % (x.real, x.imag) for x in data]
        tmp = ", ".join(tmp)
        tmp = "".join(("[", tmp, "]"))
        return tmp
    count = 0
    for d, m in zip(data, mom):
        print("% 11s = %11s + %11s => %s" % (\
                _d1(m[0]), _d1(m[1]), _d1(m[2]), _d2(d)))
        if empty is not None:
            count += 1
            if count == empty:
                count = 0
                print("")

def cg_to_pandas(data, mom):
    """Converts the Clebsch-Gordan coefficients for a two-particle operator
    into a pandas DataFrame according to the conventions used in subduction.py.
    """

    nb_mom = len(mom)
    nb_rows = len(data[0])

    # convert momenta into tuples (p_so, p_si) and copy for every row
    # fill \gamma_5 as gamma structure i. e. two pions hardcoded
    # flatten data to one list with row running faster than momentum
    df = DataFrame({'p' : [tuple([tuple(m[1]), tuple(m[2])]) \
                                       for m in mom for _i in range(nb_rows)], \
                    '\gamma' : [(5,5)] * nb_mom*nb_rows, \
                    'cg-coefficients' : 
                                     [cg for cg_row in data for cg in cg_row]}, \
                   index = pd.Index( range(nb_rows) * nb_mom, name='\mu'))
    print df


if __name__ == "__main__":
    print("for checks execute the test script")

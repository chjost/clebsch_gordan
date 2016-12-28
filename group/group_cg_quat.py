"""Class for the clebsch-gordan coefficients of a group."""

import numpy as np
import itertools as it

import group_class
import utils

class TOhCG(object):
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
        # returns None if groups is None
        self.coset1 = self.gen_coset(groups, self.p1)
        self.coset2 = self.gen_coset(groups, self.p2)
        #print(self.coset1)
        #print(self.coset2)

        # generate the allowed momentum combinations and sort them into cosets
        self.gen_momenta()
        if groups is not None:
            self.sort_momenta(groups[0])

        # calculate induced rep gamma
        # here for p1 and p2 the A1(A2) irreps are hard-coded
        # since only these contribute to pi-pi scattering
        if groups is None:
            self.gamma1 = None
            self.gamma2 = None
        else:
            irstr = "A1u" if int(p1) in [0,3] else "A2u"
            #irstr = "E1g"
            self.gamma1 = self.gen_ind_reps(groups, p1, irstr, self.coset1)
            irstr = "A1u" if int(p2) in [0,3] else "A2u"
            #irstr = "E1g"
            self.gamma2 = self.gen_ind_reps(groups, p2, irstr, self.coset2)
        #print(self.gamma1[:5])
        #print("traces of induced representation")
        #print(self.spur1)
        #print(self.spur2)

        #self.subduce()

        self.irreps = []
        self.cgs = []

    def gen_coset(self, groups, p):
        """Cosets contain the numbers of the rotation objects
        """
        if groups is None:
            return None
        g0 = groups[0]
        g1 = groups[p]
        n = int(g0.order/g1.order)
        if n == 0:
            raise RuntimeError("number of cosets is 0!")
        coset = np.zeros((n, g1.order), dtype=int)
        l = g0.order
        l1 = g1.order
        # set the subgroup as first coset
        count = 0
        for r in range(l1):
            elem = g1.lelements[r]
            if elem in g0.lelements:
                coset[0, count] = elem
                count += 1
        # calc the cosets
        uniq = np.unique(coset)
        cnum = 1 # coset number
        for elem in g0.lelements:
            if elem in uniq:
                continue
            count = 0
            for elem1 in g1.lelements:
                if elem1 in g0.lelements:
                    look = g0.lelements.index(elem1)
                    el = g0.tmult_global[elem, look]
                    coset[cnum, count] = el
                    count += 1
            cnum += 1
            uniq = np.unique(coset)
        if len(uniq) != g0.order:
            print("some elements got lost!")
        if cnum != n:
            print("some coset not found!")
        return coset

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

    def sort_momenta(self, g0):
        # check if cosets exists
        if self.coset1 is None or self.coset2 is None:
            self.smomenta1 = None
            self.smomenta2 = None
            return
        def check_coset(g0, pref, p, coset):
            res = []
            for elem in coset:
                look = g0.lelements.index(elem)
                quat = g0.elements[look]
                rvec = quat.rotation_matrix(True).dot(pref)
                c1 = utils._eq(rvec, p)
                if c1:
                    res.append(True)
                else:
                    res.append(False)
            return res
        # search for conjugacy class so that
        # R*p_ref = p
        res1 = []
        res2 = []
        for p1 in self.momenta1:
            for i,c in enumerate(self.coset1):
                t = check_coset(g0, self.pref1, p1, c)
                if np.all(t):
                    res1.append((p1, i))
                    break
        for p2 in self.momenta2:
            for i,c in enumerate(self.coset2):
                t = check_coset(g0, self.pref2, p2, c)
                if np.all(t):
                    res2.append((p2, i))
                    break
        self.smomenta1 = res1
        if len(self.smomenta1) != len(self.momenta1):
            print("some vectors not sorted, momentum 1")
        self.smomenta2 = res2
        if len(self.smomenta2) != len(self.momenta2):
            print("some vectors not sorted, momentum 2")
            print(t)

    def gen_ind_reps(self, groups, p, irstr, coset):
        g0 = groups[0]
        g1 = groups[p]
        ir = g1.irreps[g1.irrepsname.index(irstr)]
        dim = ir.dim
        ndim = (g0.order, coset.shape[0]*dim, coset.shape[0]*dim)
        gamma = np.zeros(ndim, dtype=complex)
        for ind, r in enumerate(g0.lelements):
            for cj, rj in enumerate(coset[:,0]):
                rrj = g0.tmult[r, rj]
                indj = slice(cj*dim, (cj+1)*dim)
                for ci, ri in enumerate(coset[:,0]):
                    riinv = g0.linv[ri]
                    riinvrrj = g0.tmult[riinv, rrj]
                    indi = slice(ci*dim, (ci+1)*dim)
                    if riinvrrj not in coset[0]:
                        continue
                    elem = g1.lelements.index(riinvrrj)
                    gamma[ind, indi, indj] = ir.mx[elem]
        return gamma


    def check_all_cosets(self, p, p1, p2):
        j1, j2 = None, None
        for m, j in self.smomenta1:
            if utils._eq(p1,m):
                j1 = j
                break
        if j1 is None:
            print("j1 is None")
        for m, j in self.smomenta2:
            if utils._eq(p2,m):
                j2 = j
                break
        if j2 is None:
            print("j2 is None")
        return j1, j2

    def multiplicities(self):
        multi = np.zeros((self.g.nclasses,), dtype=complex)
        for i in range(self.g.order):
            chars = np.asarray([np.trace(ir.mx[i]).conj() for ir in self.g.irreps])
            look = self.g.lelements[i]
            chars *= np.trace(self.gamma1[look])
            chars *= np.trace(self.gamma2[look])
            multi += chars
        #multi = np.real_if_close(np.rint(multi/self.g.order))
        multi = np.real_if_close(multi/self.g.order)
        return multi

    def check_index(self, mu1, mu2, dim1, dim2):
        i1 = mu1 % self.coset1.shape[0]
        p1 = None
        for m, j in self.smomenta1:
            if j == i1:
                p1 = m
                break
        if p1 is None:
            raise RuntimeError("Momentum 1 not found")

        i2 = mu2 % self.coset2.shape[0]
        p2 = None
        for m, j in self.smomenta2:
            if j == i2:
                p2 = m
                break
        if p2 is None:
            raise RuntimeError("Momentum 2 not found")
        
        p12 = p1+p2
        for m in self.momenta:
            if utils._eq(p12, m):
                return True
        return False

    def calc_cg_new(self):
        #multi = self.multiplicities()
        multi = np.zeros((self.g.nclasses,), dtype=int)
        dim1 = self.gamma1.shape[1]
        dim2 = self.gamma2.shape[1]
        #print("dim1 = %d, dim2 = %d" % (dim1, dim2))
        dim12 = dim1*dim2
        coeff = np.zeros((dim12,), dtype=complex)
        self.cgnames = []
        self.cgind = []
        self.cg = []
        lcoeffs = []
        lind = []
        for indir, ir in enumerate(self.g.irreps):
            #if np.abs(multi[indir]) < self.prec:
            #    continue
            dim = ir.dim
            dimall = dim*dim1*dim2
            ncg = np.zeros((dim,), dtype=int)
            # loop over all column index combinations
            for mup, mu1, mu2 in it.product(range(dim), range(dim1), range(dim2)):
                if not self.check_index(mu1, mu2, dim1, dim2):
                    continue
                # loop over the row of the final irrep
                for mu in range(dim):
                    indmu = dim12*mu
                    coeff.fill(0.)
                    # loop over all combinations of rows of the induced
                    # representations
                    for mu1p, mu2p in it.product(range(dim1), range(dim2)):
                        if not self.check_index(mu1p, mu2p, dim1, dim2):
                            continue
                        ind12 = dim2 * mu1p + mu2p
                        co = 0.j
                        for i in range(self.g.order):
                            tmp = ir.mx[i][mu, mup].conj()
                            look = self.g.lelements[i]
                            tmp *= self.gamma1[look, mu1p, mu1]
                            tmp *= self.gamma2[look, mu2p, mu2]
                            co += tmp
                        coeff[ind12] = co*dim
                    coeff /= self.g.order
                    ncoeff = np.sqrt(np.vdot(coeff, coeff))
                    # if norm is 0, try next combination of mu', mu1, mu2
                    if ncoeff < self.prec:
                        continue
                    else:
                        coeff /= ncoeff
                    for vec in lcoeffs:
                        coeff = utils.gram_schmidt(coeff, vec, prec=self.prec)
                        ncoeff = np.vdot(coeff, coeff)
                        # if zero vector, try next combination of mu', mu1, mu2
                        if ncoeff < self.prec:
                            break
                    if ncoeff > self.prec:
                        lcoeffs.append(coeff.copy())
                        lind.append((mu, mu1p, mu2p))
                        ncg[mu] += 1
                        multi[indir] += 1
            #for d in range(dim):
            #    if ncg[d] != multi[indir] and multi[indir] != 0:
            #        print("not the correct number of vectors found for irrep %s" % ir.name)
            #        print("found %d of %d" % (ncg[d], multi[indir]))
            self.cgnames.append((ir.name, multi[indir]/dim))
        self.cg.append(np.asarray(lcoeffs).copy())
        self.cgind.append(np.asarray(lind).copy())


    def calc_pion_cg(self, p, p1, p2, irname):
        """Calculate the elements of the Clebsch-Gordan matrix.

        Assumes that p=p1+p2, where all three are 3-vectors.
        """
        # get irrep of group g
        ir = self.g.irreps[self.g.irrepsname.index(irname)]
        # j1 and j2 are the conjugacy classes containing
        # the given momenta p1 and p2
        j1, j2 = self.check_all_cosets(p, p1, p2)
        dim = ir.dim
        dim1 = self.gamma1.shape[1]
        dim2 = self.gamma2.shape[1]
        tmp = []
        cg = np.zeros((dim,), dtype=complex)
        for mu, mu1, mu2 in it.product(range(dim), range(dim1), range(dim2)):
            #print("tmp array")
            #print(tmp)
            # reset vector
            cg.fill(0.)
            # calculate Clebsch-Gordan coefficients
            for ind, r in enumerate(self.g.lelements):
                # representation matrix for element
                rep = ir.mx[ind]
                # look up the index of the group element in the
                # g0 group
                look = self.g0.lelements.index(r)
                # hard coded for pi-pi scattering
                g1 = self.gamma1[look,j1, mu1]
                if utils._eq(g1):
                    continue
                g2 = self.gamma2[look,j2, mu2]
                if utils._eq(g2):
                    continue
                cg += rep[:,mu].conj()*g1*g2
            cg *= float(dim)/self.g.order
            # save if not zero
            tmp.append(cg.copy())
        cg = np.asarray(tmp)
        if cg.dtype == np.ndarray:
            print("cg debug")
            print(cg)
        if cg.size < 1:
            cg = None
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
        if result.size < 1 or utils._eq(result):
            cgs = None
        else:
            # orthonormalize the basis
            cgs = self._norm_cgs(result)
        self.irreps.append(irname)
        self.cgs.append(cgs)
        return irname, cgs, self.allmomenta

    def _norm_cgs(self, data):
        if data.ndim > 1:
            _data = data
        else:
            _data = data.reshape(-1,1)
        # prepare result array
        res = np.zeros(_data.shape, dtype=complex)
        # sort by final momentum, so that all final momenta are
        # normalized seperately
        ind = [[] for x in self.momenta]
        for i, m in enumerate(self.allmomenta):
            for j, fm in enumerate(self.momenta):
                if np.array_equal(m[0], fm):
                    ind[j].append(i)
                    break
        print(ind)
        # norm the data
        # set starting variables
        for i in range(_data.shape[1]):
            for j in ind:
                tmp = _data[j,i]
                if np.any(tmp):
                    norm = np.sqrt(np.vdot(tmp, tmp))
                    res[j,i] = tmp/norm
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

if __name__ == "__main__":
    print("for checks execute the test script")

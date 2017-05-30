"""Class for the clebsch-gordan coefficients of a group."""

import numpy as np
import itertools as it

try:
    import pandas as pd
    from pandas import Series, DataFrame
    usepandas=True
except ImportError:
    usepandas=False
try:
    import sympy
    usesympy=True
except ImportError:
    usesympy=False

import group_class
import utils
import latex_utils as lu

class TOhCG(object):
    def __init__(self, p, p1, p2, groups=None, ir1=None, ir2=None):
        """p, p1, and p2 are the magnitudes of the momenta.
        """
        self.prec = 1e-6
        # save the norm of the momenta for the combined system
        # and each particle
        self.p = p
        self.p1 = p1
        self.p2 = p2
        indp0, indp1, indp2, indp = None, None, None, None
        self.U0 = None
        # transformation from cartesian to symmetrized spherical
        # harmonics
        self._U = np.asarray([[0.,-1.j,0.],[0.,0.,1.],[-1.,0.,0.]])
        if groups is not None:
            pindex = [x.p2 for x in groups]
            try:
                # octohedral group always needed (at least in induction step).
                indp0 = pindex.index(0)
                # pick all groups from groups which belong to p, p1 or p2 respectively. 
                # If no group for a given p is in groups, raise an RuntimeError
                indp = pindex.index(p)
                indp1 = pindex.index(p1)
                indp2 = pindex.index(p2)
            except IndexError:
                raise RuntimeError("no group with P^2 = %d (%d, %d) found" % (p, p1, p2))

            # In TOh, the standard basis are symmetrized spherical harmonics. 
            # If another basis is chosen, the basis transformation matrix 
            # member is nontrivial, otherwise it contains the unit matrix
            self.U0 = groups[indp0].U3
            self.T0 = groups[indp0].U2
            if (not utils._eq(self.U0, groups[indp].U3)) or (
                    not utils._eq(self.T0, groups[indp].U2)):
                raise RuntimeError("The group projecting to has different basis")
            if (not utils._eq(self.U0, groups[indp1].U3)) or (
                    not utils._eq(self.T0, groups[indp1].U2)):
                raise RuntimeError("The group of op 1 has different basis")
            if (not utils._eq(self.U0, groups[indp1].U3)) or (
                    not utils._eq(self.T0, groups[indp1].U2)):
                raise RuntimeError("The group of op 2 has different basis")

            # Succesively transform from cartesian coordinates (momenta) to 
            # basis chosen in group
            self._U = self.U0.dot(self._U)

            # save reference momenta in group basis
            self.pref = self._U.dot(groups[indp0].pref_cart)
            self.pref1 = self._U.dot(groups[indp1].pref_cart)
            self.pref2 = self._U.dot(groups[indp2].pref_cart)

        # get the cosets, always in the maximal group (2O here)
        # returns None if groups is None
        self.coset1 = self.gen_coset(groups, indp0, indp1)
        self.coset2 = self.gen_coset(groups, indp0, indp2)
        #print(self.coset1)
        #print(self.coset2)

        # generate the allowed momentum combinations and sort them into cosets
        self.gen_momenta(pm=max(p, p1, p2, 4))

        # calculate induced rep gamma
        # here for p1 and p2 the A1(A2) irreps are hard-coded
        # since only these contribute to pi-pi scattering
        if groups is None:
            self.gamma1 = None
            self.gamma2 = None
            self.dim1 = 0
            self.dim2 = 0
            self.irstr1 = ir1
            self.irstr2 = ir2
        else:
            if ir1 is None:
                self.irstr1 = "A1u" if int(p1) in [0,3,5,6] else "A2u"
            else:
                self.irstr1 = ir1
            if self.irstr1 not in groups[indp1].irrepsname:
                raise RuntimeError("irrep %s not in group 1!" % self.irstr1)
            else:
                self.dim1 = groups[indp1].irrepsname.index(self.irstr1)
                self.dim1 = groups[indp1].irrepdim[self.dim1]
            self.gamma1 = self.gen_ind_reps(groups, indp0, indp1,
                                self.irstr1, self.coset1)
            #print(self.gamma1[:5])

            if ir2 is None:
                self.irstr2 = "A1u" if int(p2) in [0,3,5,6] else "A2u"
            else:
                self.irstr2 = ir2
            if self.irstr2 not in groups[indp2].irrepsname:
                raise RuntimeError("irrep %s not in group 2!" % self.irstr2)
            else:
                self.dim2 = groups[indp2].irrepsname.index(self.irstr2)
                self.dim2 = groups[indp2].irrepdim[self.dim2]
            self.gamma2 = self.gen_ind_reps(groups, indp0, indp2,
                                self.irstr2, self.coset2)
            #print(self.gamma2[:5])

            self.sort_momenta(groups[indp0])

        self.calc_cg_ha(groups, indp)
        #self.calc_cg_new(groups, indp)
        #self.check_cg()

    @classmethod
    def read(cls, path=None, filename=None, momenta=(0,0,0), irreps=["A1u","A1u"]):
        if filename is None:
            _filename = "CG_P%02d_k%02d_%s_k%02d_%s.npz" % (
                    momenta[0], momenta[1], irreps[0], momenta[2], irreps[1])
        else:
            _filename = filename
        if path is None:
            _path = "./cg/"
        else:
            _path = path
        _name = "/".join([_path, _filename])

        fh = np.load(_name)
        params = fh["params"]
        p, p1, p2 = params[0]
        cgnames = [x for x in params[6:]]
        tmp = cls(p, p1, p2)
        tmp.cgnames = cgnames
        tmp.irstr1, tmp.dim1, tmp.irstr2, tmp.dim2 = params[1]
        tmp.indices = params[2]
        tmp.smomenta1 = params[3]
        tmp.smomenta2 = params[4]
        tmp.allmomenta = params[5]
        tmp.cg = fh["cg"]
        tmp.cgind = fh["cgind"]
        tmp.coset1 = fh["coset1"]
        tmp.coset2 = fh["coset2"]
        tmp.gamma1 = fh["gamma1"]
        tmp.gamma2 = fh["gamma2"]
        Umat = fh["Umat"]
        tmp.U0 = Umat[0]
        tmp._U = Umat[1]
        del fh
        return tmp

    def save(self, path=None, filename=None):
        if filename is None:
            _filename = "CG_P%02d_k%02d_%s_k%02d_%s.npz" % (
                    self.p, self.p1, self.irstr1, self.p2, self.irstr2)
        else:
            _filename = filename
        if path is None:
            _path = "./cg/"
        else:
            _path = path
        _name = "/".join([_path, _filename])
        utils.ensure_write(_name)
        # TODO ensure path
        params = []
        params.append((self.p, self.p1, self.p2))
        params.append((self.irstr1, self.dim1, self.irstr2, self.dim2))
        params.append(self.indices)
        params.append(self.smomenta1)
        params.append(self.smomenta2)
        params.append(self.allmomenta)
        for cgn in self.cgnames:
            params.append(cgn)
        params = np.asarray(params, dtype=object)
        Umat = np.asarray([self.U0, self._U])
        np.savez(_name, params=params, cg=self.cg, cgind=self.cgind,
                gamma1=self.gamma1, gamma2=self.gamma2,
                coset1=self.coset1, coset2=self.coset2,
                Umat=Umat)

    def gen_coset(self, groups, p0, p):
        """Cosets contain the numbers of the rotation objects
        """
        if groups is None:
            return None
        g0 = groups[p0]
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

    def gen_momenta(self, pm=4):
        #print("pm=%d" % pm)
        # pm: maximum component in each direction
        def _abs(x):
            return np.vdot(x, x) <= pm
        def _abs1(x,a):
            return np.vdot(x, x) == a
        gen = it.ifilter(_abs, it.product(range(-pm,pm+1), repeat=3))
        lp3 = [np.asarray(y, dtype=int) for y in gen]
        self.momenta =  [self._U.dot(y) for y in it.ifilter(lambda x: _abs1(x,self.p), lp3)]
        self.momenta1 = [self._U.dot(y) for y in it.ifilter(lambda x: _abs1(x,self.p1), lp3)]
        self.momenta2 = [self._U.dot(y) for y in it.ifilter(lambda x: _abs1(x,self.p2), lp3)]
        # check allowed momentum combinations
        self.allmomenta = []
        for p in self.momenta:
            for p1 in self.momenta1:
                for p2 in self.momenta2:
                    if utils._eq(p1+p2-p):
                        self.allmomenta.append((p, p1, p2))
        if not self.allmomenta:
            raise RuntimeError("no valid momentum combination found")

    def sort_momenta(self, g0):
        # check if cosets exists
        if self.coset1 is None or self.coset2 is None:
            print("some coset does not exist")
            self.smomenta1 = None
            self.smomenta2 = None
            return
        T1irrep = g0.irreps[g0.irrepsname.index("T1u")]
        def check_coset(g0, pref, p, coset):
            res = []
            # check needs to be done in basis of T1u
            for elem in coset:
                look = g0.lelements.index(elem)
                rvec = T1irrep.mx[look].dot(pref)
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
        # generate lists with all allowed momenta combinations and their
        # combined coset+dim indices
        dimcomb = [x for x in it.product(range(self.dim1), range(self.dim2))]
        self.indices = []
        for p, p1, p2 in self.allmomenta:
            j1, j2 = self.check_all_cosets(p, p1, p2)
            for x in dimcomb:
                self.indices.append((j1*self.dim1+x[0], j2*self.dim2+x[1]))

    def gen_ind_reps(self, groups, p0, p, irstr, coset):
        g0 = groups[p0]
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

    def multiplicities(self, groups, p):
        g = groups[p]
        multi = np.zeros((g.nclasses,), dtype=complex)
        for i in range(g.order):
            chars = np.asarray([np.trace(ir.mx[i]).conj() for ir in g.irreps])
            look = g.lelements[i]
            tr1 = tr2 = 0.
            for i1, i2 in self.indices:
                if i1 != i2:
                    continue
                tr1 += self.gamma1[look,i1,i1]
                tr2 += self.gamma2[look,i2,i2]
            chars *= tr1*tr2
            #chars *= np.trace(self.gamma1[look])
            #chars *= np.trace(self.gamma2[look])
            multi += chars
        multi = np.real_if_close(multi/g.order)
        return multi

    def calc_cg_ha(self, groups, p):
        self.cgnames = []
        self.cgind = []
        self.cg = []
        if groups is None:
            return
        g = None
        # get the group associated with the momentum squared p
        for _g in groups:
            if _g.p2 == p:
                g = _g
                break
        if g is None:
            raise RuntimeError("no group for found for calculating CG's.")
        multi = 0
        dim1 = self.gamma1.shape[1]
        dim2 = self.gamma2.shape[1]
        dim12 = dim1*dim2
        coeff = np.zeros((len(self.indices), len(self.indices)), dtype=complex)
        # function for looping over all group elements
        def all_cg(i,j,k,l,m,n):
            res = 0.
            for ind in range(g.order):
                tmp = ir.mx[ind][i,j].conj()
                look = g.lelements[ind]
                tmp *= self.gamma1[look, k, l]
                tmp *= self.gamma2[look, m, n]
                res += tmp
            return res
        for indir, ir in enumerate(g.irreps):
            m1, m2, m3 = None, None, None
            multi = 0
            lind = []
            lcoeffs = []
            dim = ir.dim
            # loop over all column index combinations that conserve the COM momentum
            for mup, (mu1, mu2) in it.product(range(dim), self.indices):
                cg = all_cg(mup, mup, mu1, mu1, mu2, mu2)
                if cg < self.prec:
                    continue
                cg *= float(dim)/g.order
                _cg = np.sqrt(cg)
                m1 = mup
                m2 = mu1
                m3 = mu2
                coeff.fill(0.)
                _check = 0.
                for mu in range(dim):
                    for ind, (mu1, mu2) in enumerate(self.indices):
                        coeff[mu, ind] = all_cg(m1, mu, m2, mu1, m3, mu2)
                    coeff[mu] *= float(dim)/g.order/_cg
                    coeff[mu] = coeff[mu].conj()
                    if mu == 0 and multi > 0:
                        #print("checking for orthogonality")
                        for m in range(multi):
                            _check = np.absolute(np.vdot(coeff[0], lcoeffs[m][0]))
                            if _check > self.prec:
                                break
                    if _check > self.prec:
                        break
                if _check > self.prec:
                    continue
                lcoeffs.append(coeff[:dim].copy())
                lind.append((m1, m2, m3))
                multi += 1

            if multi > 0:
                self.cgnames.append((ir.name, multi, dim))
                self.cg.append(np.asarray(lcoeffs).copy())
                self.cgind.append(np.asarray(lind).copy())
        # for easier storage, save in one big numpy array with
        # dimensions [irrep index, max mult, max dim, # of coefficients]
        nirreps = len(self.cg)
        mmult = max([x.shape[0] for x in self.cg])
        mdim = max([x.shape[1] for x in self.cg])
        nc = self.cg[0].shape[2]
        newcg = np.zeros((nirreps, mmult, mdim, nc), dtype=complex)
        newind = np.zeros((nirreps, mmult, 3), dtype=int)
        for iind, _cg in enumerate(self.cg):
            d = _cg.shape
            newcg[iind,:d[0],:d[1]] = _cg
            newind[iind,:d[0]] = self.cgind[iind]

        self.cg = newcg
        self.cgind = newind

    def calc_cg_new_(self, groups, p):
        self.cgnames = []
        self.cgind = []
        self.cg = []
        if groups is None:
            return
        g = groups[p]
        multi = 0
        dim1 = self.gamma1.shape[1]
        dim2 = self.gamma2.shape[1]
        dim12 = dim1*dim2
        coeff = np.zeros((len(self.indices),), dtype=complex)
        lind = []
        for indir, ir in enumerate(g.irreps):
            multi = 0
            lcoeffs = []
            dim = ir.dim
            mup = 0
            # loop over all column index combinations that conserve the COM momentum
            for mu1, mu2 in self.indices:
            #for mup, (mu1, mu2) in it.product(range(dim), self.indices):
                # loop over the row of the final irrep
                for mu in range(dim):
                    if mu != mup:
                        continue
                    coeff.fill(0.)
                    # loop over all combinations of rows of the induced
                    # representations
                    for ind1, (mu1p, mu2p) in enumerate(self.indices):
                        for i in range(g.order):
                            #tmp = ir.mx[i][mu, mu].conj()
                            tmp = ir.mx[i][mu, mup].conj()
                            look = g.lelements[i]
                            tmp *= self.gamma1[look, mu1p, mu1]
                            tmp *= self.gamma2[look, mu2p, mu2]
                            coeff[ind1] += tmp
                    coeff *= float(dim)/g.order
                    ncoeff = np.sqrt(np.vdot(coeff, coeff))
                    # if norm is 0, try next combination of mu', mu1, mu2
                    if ncoeff < self.prec:
                        continue
                    else:
                        coeff /= ncoeff
                    # orthogonalize w.r.t. already found vectors of same irrep
                    for vec in lcoeffs:
                        coeff = utils.gram_schmidt(coeff, vec, prec=self.prec)
                        ncoeff = np.sqrt(np.vdot(coeff, coeff))
                        # if zero vector, try next combination of mu', mu1, mu2
                        if ncoeff < self.prec:
                            break
                    if ncoeff < self.prec:
                        continue
                    # orthogonalize w.r.t. already found vectors of other irreps
                    for lcg in self.cg:
                        for vec in lcg:
                            coeff = utils.gram_schmidt(coeff, vec, prec=self.prec)
                            ncoeff = np.sqrt(np.vdot(coeff, coeff))
                            # if zero vector, try next combination of mu', mu1, mu2
                            if ncoeff < self.prec:
                                break
                        if ncoeff < self.prec:
                            break
                    if ncoeff > self.prec:
                        lcoeffs.append(coeff.copy())
                        lind.append((mu, mup, mu1, mu2))
                        multi += 1
            if multi > 0:
                print("%s: %d times" % (ir.name, multi))
                self.cgnames.append((ir.name, multi, dim))
                self.cg.append(np.asarray(lcoeffs).copy())
                self.cgind.append(np.asarray(lind).copy())
        #print("before saving as array")
        #print(self.cg)
        self.cg = np.asarray(self.cg)
        #print("after saving as array")
        #print(self.cg)
        self.cgind = np.asarray(self.cgind)

    def check_cg(self):
        """Rows of the Clebsch-Gordan coefficients are orthonormal by construction.
        This routine checks the columns of the Clebsch-Gordan coefficients."""
        #if isinstance(self.cg, list):
        #    return
        #print(self.cg)
        allcgs = []
        for lcg in self.cg:
            for vec in lcg:
                allcgs.append(vec)
        allcgs = np.asarray(allcgs)
        print(allcgs)
        ortho = True
        norm = True
        for i, j in it.combinations(range(allcgs.shape[1]), 2):
            test = np.sqrt(np.vdot(allcgs[:,i], allcgs[:,j]))
            if test > self.prec:
                #print("columns %d and %d not orthogonal" % (i, j))
                ortho = False
        for i in range(allcgs.shape[1]):
            test = np.sqrt(np.vdot(allcgs[:,i], allcgs[:,i]))
            if np.abs(test-1.) > self.prec:
                #print("columns %d not normalized" % (i))
                norm = False
        if ortho is False:
            print("Some clebsch-gordan vector not orthogonalized.")
        if norm is False:
            print("Some clebsch-gordan vector not normalized.")

    def display(self, emptyline=None):
        def tostring(data):
            tmp = ["%+.3f%+.3fj" % (x.real, x.imag) for x in data]
            tmp = ",".join(tmp)
            tmp = "".join(("(", tmp, ")"))
            return tmp
        def momtostring(data):
            tmp = []
            for d in data:
                _d = np.real_if_close((self._U.conj().T).dot(d))
                tmp.append("[%+.0f,%+.0f,%+.0f]" % (_d[0], _d[1], _d[2]))
            tmpstr = "%s x %s -> %s" % (tmp[1], tmp[2], tmp[0])
            return tmpstr

        count = 0
        dim1 = self.gamma1.shape[1]
        dim2 = self.gamma2.shape[1]
        # loop over irreps
        for i, (name, multi, dim) in enumerate(self.cgnames):
            if multi < 1:
                continue
            print(" %s ".center(20,"*") % name)
            # loop over multiplicities
            for m in range(multi):
                print("multiplicity %d" % m)
                for ind, (p, p1, p2) in enumerate(self.allmomenta):
                    data = self.cg[i,m,:dim,ind]
                    tmpstr = "%s: %s" % (momtostring([p,p1,p2]), tostring(data))
                    print(tmpstr)
                    if emptyline is not None:
                        count += 1
                        if count == emptyline:
                            print("")
                            count = 0

    def tables(self):
        def tostring(data):
            tmp = ["%+.3f%+.3fj" % (x.real, x.imag) for x in data]
            tmp = ",".join(tmp)
            tmp = "".join(("(", tmp, ")"))
            return tmp
        # loop over irreps
        for i, (name, multi, dim) in enumerate(self.cgnames):
            if multi < 1:
                continue
            print(" %s ".center(20,"*") % name)
            # loop over multiplicities
            for m in range(multi):
                print("multiplicity %d" % m)
                data = np.real_if_close(self.cg[i][m])
                for ind, d in enumerate(np.atleast_2d(data[:dim])):
                    tmpstr = ["%.3f%+.3fj" % (x.real, x.imag) for x in d]
                    tmpstr = ", ".join(tmpstr)
                    tmpstr = "%d: %s" % (ind, tmpstr)
                    print(tmpstr)

    def get_cg(self, p1, p2, irrep, change_basis=True):
        """Pass the 3-momenta of both particles,
        check for correct order of momenta.
        """
        # change the basis to the one used
        if change_basis:
            _p1 = self._U.dot(p1)
            _p2 = self._U.dot(p2)
        else:
            _p1 = np.asarray(p1)
            _p2 = np.asarray(p2)
        cg = []
        index = None
        # select correct momentum
        for ind, (p, k1, k2) in enumerate(self.allmomenta):
            if utils._eq(k1, _p1) and utils._eq(k2, _p2):
                index = ind
                break
        # select correct momentum
        #for ind, (p, k1, k2) in enumerate(self.allmomenta):
        #    if utils._eq(k1, p1) and utils._eq(k2, p2):
        #        index = ind
        #        break
        # if momentum not allowed, return None
        if index is None:
            #print("Momentum not present!")
            return None

        # get coefficients
        for i, (name, multi, dim) in enumerate(self.cgnames):
            if irrep != name:
                continue
            #print(self.cg.shape)
            cg = self.cg[i,:multi,:dim,index]
            return cg
        return None

    def print_operators(self, width=None):
        def momtostring(p):
            _p = np.real_if_close((self._U.conj().T).dot(p))
            return " (%+1d,%+1d,%+1d)" % (_p[0],_p[1],_p[2])
        def cgtostring(cg):
            _n = np.absolute(cg) < self.prec
            _r = np.absolute(cg.real) < self.prec
            _i = np.absolute(cg.imag) < self.prec
            if _n:
                return "%11d" % 0
            elif _r and not _i:
                return "%+ 10.3fi" % cg.imag
            elif not _r and _i:
                return "%+ 10.3f " % cg.real
            else:
                return "%+.2f%+.2fi" % (cg.real, cg.imag)
        # set width for better screen printing
        _l = len(self.allmomenta)
        if width is None:
            if _l < 9:
                _w = _l
            else:
                _w = 6
        else:
            _w = width
        n = _l//_w

        line = "-" * (5+_w*12)
        print("O(p_1) x O(p_2) -> O(P)")
        for _n in range(n):
            # print momentum header
            s, s1, s2 = [], [], []
            _s = slice(_n*_w, (_n+1)*_w)
            for p, p1, p2 in self.allmomenta[_s]:
                s1.append(momtostring(p1))
                s2.append(momtostring(p2))
                s.append(momtostring(p))
            s1 = "|".join(s1)
            s1 = "|".join([" p1 ", s1])
            s2 = "|".join(s2)
            s2 = "|".join([" p2 ", s2])
            s = "|".join(s)
            s = "|".join([" P  ", s])
            print(s1)
            print(s2)
            print(s)
            print(line)

            # print the irreps
            for i, (name, multi, dim) in enumerate(self.cgnames):
                tmp = "%4s" % name
                for m in range(multi):
                    for d in range(dim):
                        cgs = [cgtostring(x) for x in self.cg[i, m, d, _s]]
                        cgs = "|".join(cgs)
                        cgs = "|".join([tmp, cgs])
                        print(cgs)
                print(line)
        m = _l % _w
        if m != 0:
            line = "-" * (5+m*12)
            # print momentum header
            s, s1, s2 = [], [], []
            _s = slice(m*_w, None)
            for p, p1, p2 in self.allmomenta[_s]:
                s1.append(momtostring(p1))
                s2.append(momtostring(p2))
                s.append(momtostring(p))
            s1 = "|".join(s1)
            s1 = "|".join([" p1 ", s1])
            s2 = "|".join(s2)
            s2 = "|".join([" p2 ", s2])
            s = "|".join(s)
            s = "|".join([" P  ", s])
            print(s1)
            print(s2)
            print(s)
            print(line)

            # print the irreps
            for i, (name, multi, dim) in enumerate(self.cgnames):
                tmp = "%4s" % name
                for m in range(multi):
                    for d in range(dim):
                        cgs = [cgtostring(x) for x in self.cg[i, m, d, _s]]
                        cgs = "|".join(cgs)
                        cgs = "|".join([tmp, cgs])
                        print(cgs)
                print(line)


    def to_latex(self, document=True, table=True, booktabs=True):
        def cgtostring(vec):
            tmpstr = []
            for c in vec:
                if np.absolute(c) < self.prec:
                    tmpstr.append(" ")
                    continue
                tmpstr.append("$%s$" % (lu.latexify(c)))
            tmpstr = " & ".join(tmpstr)
            return tmpstr
        if document:
            packages = ["amsmath"]
            if booktabs:
                packages.append("booktabs")
            lu.start_document(packages)
        if document or table:
            a = "c" * len(self.allmomenta)
            align = "".join(["lcc|", a])
            lu.start_table(align=align)
        if booktabs:
            print("\\toprule")
        add = "& " * (len(self.allmomenta))
        print("Irrep & row & multiplicity %s \\\\" % add)
        line = False # flag if print line
        trule, mrule, brule = lu.hrules(booktabs)
        print(trule)
        for i, (name, multi, dim) in enumerate(self.cgnames):
            if line:
                print(mrule)
                line = False
            for m in range(multi):
                for d in range(dim):
                    tstr = cgtostring(self.cg[i,m,d])
                    tmp = ["%s"%name, "%d"%d, "%d"%m, tstr]
                    tmp = " & ".join(tmp)
                    tmp = " ".join([tmp, "\\\\"])
                    print(tmp)
                    line = True
        if line:
            print(brule)
        if document or table:
            lu.end_table()
        if document:
            lu.end_document()

    def to_pandas(self, irrep=None):
        if not usesympy:
            print("SymPy not available")
            return
        elif not usepandas:
            print("Pandas is not available")
            return
        # sympify the coefficients
        def _s(x):
            tmp = sympy.nsimplify(x)
            tmp1 = sympy.simplify(tmp)
            # TODO: does no align properly using rjust
            # maybe due to implicit string conversion
            return str(tmp1)

        # prepare some constants
        plists = [[], [], []]
        for p, p1, p2 in self.allmomenta:
            plists[0].append(p)
            plists[1].append(p1)
            plists[2].append(p2)
        psize = len(self.allmomenta)
        dfdict = {"Irrep" : [],\
                  "cg" : [],\
                  "row" : [],\
                  "multi" : [],\
                  "ptot": [],\
                  "p1" : [],\
                  "p2" : []}
        # actual work
        for i, (name, multi, dim) in enumerate(self.cgnames):
            # skip if irrep is given and not the same
            if irrep is not None and name != irrep:
                continue
            size = multi*dim
            tsize = size*psize

            dfdict["Irrep"] = dfdict["Irrep"] + [name] * tsize
            dfdict["ptot"] = dfdict["ptot"] + plists[0] * size
            dfdict["p1"] = dfdict["p1"] + plists[1] * size
            dfdict["p2"] = dfdict["p2"] + plists[2] * size
            for m in range(multi):
                for d in range(dim):
                    dfdict["row"] = dfdict["row"] + [d+1]*psize
                    dfdict["multi"] = dfdict["multi"] + [m+1]*psize
                    dfdict["cg"] = dfdict["cg"] + [\
                            _s(x) for x in self.cg[i,m,d]]

        df = DataFrame(dfdict)
#        print(df)
        return df

if __name__ == "__main__":
    print("for checks execute the test script")

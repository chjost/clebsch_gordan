"""Class for the basis of a group."""

import numpy as np

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

import quat
import utils
import latex_utils as lu


# Generate basis vectors of irreducible representation i.e. subduction coefficients
# jmax: Maximum spin +1 for which bases are constructed. Always have to construct 
# lower j because construction works iteratively
class TOhBasis(object):
    def __init__(self, group=None, multi=1, prec=1e-6, verbose=1, jmax=None):
        self.prec = prec
        self.verbose = verbose
        # check if group is given and has everything needed
        if group is None:
            return # empty object
        try:
            _ = group.irreps
        except AttributeError:
            raise RuntimeError("group has no irreps to calculate basis of")
        # check further arguments
        if multi < 1:
            print("multiplicity < 1 does not make sense, setting to 1")
            _multi = 1
        else:
            _multi = multi
        if jmax is None:
            self.maxj = self.get_max_j(group, _multi)
        else:
            print("max j found, overwritting multi")
            self.maxj = min(50, jmax) # hardcoded upper limit

        # save some needed parameters
        self.irrepsname = group.irrepsname
        self.dims = group.irrepdim
        _pref = group.pref_cart
        self.pref = (0,0,0) if _pref is None else tuple(_pref)

        # Everything in group is done in symmetrized spherical harmonics
        # according to Altmann & Herzig, Point-group Theory Tables, 2011.
        # The following basis change is done on top for the [2-4]D groups,
        # respectively.
        self.U2 = group.U2
        self.U3 = group.U3
        self.U4 = group.U4

        # calculate rotation matrices in 2j+1 dimensions up to maxj
        self.Rmat = self.precalc_R_matrices(group)

        self.basis, self.multi = self.calculate(group)

    def calculate(self, group):
        basis = []
        bmulti = np.zeros((self.maxj, len(self.dims)), dtype=int)
        for j in range(self.maxj):
            basis.append([])
            num = self.multiplicityO3(group, j)
            for i, (ir, n) in enumerate(zip(group.irreps, num)):
                # if irrep does not contribute to j, continue
                if n < 1:
                    basis[-1].append(None)
                    continue
                # else calculate basis vectors
                bvec = self.calc_basis_vec(ir, j, self.dims[i], oldbase=basis[-1])
                basis[-1].append(utils.clean_complex(bvec))
                bmulti[j][i] += int(bvec.shape[0])//int(self.dims[i])
        return basis, bmulti

    def get_max_j(self, group, multi):
        line = "".center(5*len(group.irreps),"-")
        def _p(m, m1, m0):
            tmpstr = ["%4s|" % n for n in group.irrepsname]
            tmpstr = "".join(tmpstr)
            print(tmpstr)
            print(line)
            _m = m-m1
            tmpstr = ["%4d|" % n for n in _m]
            tmpstr = "".join(tmpstr)
            print(tmpstr)
            print(line)
            _m = m-m0
            tmpstr = ["%4d|" % n for n in _m]
            tmpstr = "".join(tmpstr)
            print(tmpstr)
            print(line)

        run = True
        m = np.zeros((len(group.irreps),),dtype=int)
        for irn in ["E1g", "E1u", "E2g", "E2u", "F1g", "F1u"]:
            try:
                ind = group.irrepsname.index(irn)
                m[ind] = multi
            except ValueError:
                pass
        #for i, ir in enumerate(group.irrepsname):
        #    if ir.endswith("u"):
        #        m[i] = multi
        j = 0
        m0 = m.copy()
        m1 = m.copy()
        while run:
            _m = self.multiplicityO3(group, j)
            m += np.asarray(_m)
            #print("j=%d" % j)
            #_p(m,m1, m0)
            if np.all(m>= multi):
                return j
            j += 1
            m1 = m.copy()
            if j == 51:
                print("stoping at j=50")
                return 50

    def calc_basis_vec(self, irrep, j, dim, oldbase=[]):
        if j >= self.maxj:
            raise RuntimeError("j cannot be calculated")
        def _check(vec, base):
            for b in base:
                vec = utils.gram_schmidt(vec, b)
                norm = np.sqrt(np.vdot(vec, vec)) 
                if norm < self.prec:
                    return None
            return vec
        base = []
        # S(J; IRREP, mu) \propto \sum_{G} T^IRREP_{mu,nu} \sum_{n} R^J_{m,n}
        for nu in range(dim):
            # find non-zero element with indices:
            # mu = nu
            # m = n
            _v = self.get_basis_vecs(irrep, nu, nu, j)
            n = -1
            for n in range(_v.shape[0]):
                if np.absolute(_v[n,n]) < self.prec:
                    continue
                for mu in range(dim):
                    vec = self.get_basis_vecs(irrep, mu, nu, j, n=n)
                    norm = np.sqrt(np.vdot(vec, vec))
                    if norm < self.prec:
                        continue
                    vec /= norm
                    # check against the already found vectors
                    vec = _check(vec, base)
                    if vec is None:
                        continue
                    # check orthogonality to all other vectors of same j
                    for ovecs in oldbase:
                        if ovecs is None:
                            continue
                        vec = _check(vec, ovecs)
                        if vec is None:
                            break
                    if vec is None:
                        continue
                    base.append(vec)
        base = np.asarray(base)
        return base

    # effectively apply a projection operator to irrep
    # yields all possible basis vectors
    def get_basis_vecs(self, irrep, row, col, j, n=None):
        prefac = float(irrep.dim)/irrep.mx.shape[0]
        jmult = int(2*j+1)
        res = np.zeros((jmult, jmult), dtype=complex)
        for k, r_mat in enumerate(self.Rmat[j]):
            coeff = irrep.mx[k,row,col].conj()
            res += coeff * r_mat
        if n is None:
            return prefac * res
        else:
            return (prefac * res)[:,n]

    def precalc_R_matrices(self, group):
        #print("start precalc")
        Rmat = []
        for j in range(self.maxj):
            tmp = []
            for q in group.elements:
                tmp.append(q.R_matrix(j))
            Rmat.append(np.asarray(tmp))
        return Rmat
        #print("end precalc")

    def multiplicityO3(self, group, j):
        # get the rotation angles
        omegas = [group.elements[a].omega() for a in group.crep]
        omegas = np.asarray(omegas)
        par = [group.elements[a].i for a in group.crep]
        par = np.asarray(par)

        # subduce spin j
        characters = np.zeros_like(omegas, dtype=complex)
        for k in range(int(2*j+1)):
            characters += np.exp(1.j*(j-k)*omegas)
        characters = utils.clean_complex(characters, self.prec)
        if j%2:
            characters *= par

        # calculate multiplicities,
        # characters already contain the inversion, if enabled
        multi = group.tchar.dot(characters*group.cdim)
        multi = np.real_if_close(np.rint(multi/float(group.order)))

        # check and yield the irrep
        check = multi.dot(group.tchar)
        if np.any((check-characters) > self.prec):
            print("characters:")
            print(characters)
            print("multiplicities:")
            print(multi)
            print("check = %r" % check)
            raise RuntimeError("subduction does not work!")
        return multi

    def print_overview(self):
        l = len(self.irrepsname)
        head = "|".join(["%4s" % x for x in self.irrepsname])
        head = "|".join(["  J", head])
        line = "-" * (4+5*l)
        print(head)
        print(line)
        for j, multis in enumerate(self.multi):
            tmp = "|".join(["%4d" % x for x in multis])
            tmp = "|".join(["%3d" % j, tmp])
            print(tmp)
            print(line)

    def print_table(self):
        def tostring(vec, j):
            tmpstr = []
            for m, c in enumerate(vec):
                if np.absolute(c) < self.prec:
                    continue
                cstr = ""
                if np.absolute(c.real) > self.prec and np.absolute(c.imag) < self.prec:
                    cstr = "%+.3f" % c.real
                elif np.absolute(c.real) < self.prec and np.absolute(c.imag) > self.prec:
                    cstr = "%+.3fi" % c.imag
                else:
                    cstr = "%+.3f%+.3fi" % (c.real, c.imag)
                tmpstr.append("%s |%d %d>" % (cstr, j, m-j))
            tmpstr = " ".join(tmpstr)
            return tmpstr

        for j, base in enumerate(self.basis):
            print("J = %d" % j)
            for ir, basevecs in enumerate(base):
                if basevecs is None:
                    continue
                m = basevecs.shape[0] // self.dims[ir]
                for ind, vec in enumerate(basevecs):
                    a, b = divmod(ind, m)
                    tstr = tostring(vec, j)
                    print("Irrep %s, row %d, mul %d: %s" % (self.irrepsname[ir], a, b, tstr))

    def to_latex(self, document=True, table=True, booktabs=True):
        def tostring(vec, j):
            tmpstr = []
            for m, c in enumerate(vec):
                if np.absolute(c) < self.prec:
                    continue
                tmpstr.append("%s |%d,%+d\\rangle" % (lu.latexify(c), j, m-j))
            tmpstr = " + ".join(tmpstr)
            tmpstr = " ".join(["$", tmpstr, "$"])
            return tmpstr
        if document:
            packages = ["amsmath"]
            if booktabs:
                packages.append("booktabs")
            lu.start_document(packages)
        if document or table:
            lu.start_table(align="l|lcc|l")
        if booktabs:
            print("\\toprule")
        print("J & Irrep & row & multiplicity & subduction coefficient \\\\")
        line = False # flag if print line
        trule, mrule, brule = lu.hrules(booktabs)
        print(trule)
        for j, base in enumerate(self.basis):
            if line:
                print(mrule)
                line = False
            for ir, basevecs in enumerate(base):
                if basevecs is None:
                    continue
                m = basevecs.shape[0] // self.dims[ir]
                for ind, vec in enumerate(basevecs):
                    a, b = divmod(ind, m)
                    tstr = tostring(vec, j)
                    tmp = ["%d"%j, self.irrepsname[ir], "%d"%a, "%d"%b, tstr]
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

    def to_pandas(self, j):
        if not usesympy:
            print("SymPy not available")
            return
        elif not usepandas:
            print("Pandas is not available")
            return
        elif j >= self.maxj:
            print("J=%d not calculated!" % j)
            return
        # sympify the coefficients
        def _s(x):
            tmp = sympy.nsimplify(x)
            tmp1 = sympy.simplify(tmp)
            # TODO: does no align properly using rjust
            # maybe due to implicit string conversion
            return str(tmp1)

        # actual work
        jmult = int(2*j+1)
        base = self.basis[j]
        df_base = DataFrame()
#        df_base.columns = pd.Index(["p", "J", "M", "coeff", "Irrep", "mult", "row"])
        for ir, basevecs in enumerate(base):
            if basevecs is None:
                continue
            iname = self.irrepsname[ir]
            m = basevecs.shape[0] // self.dims[ir]
            for ind, vec in enumerate(basevecs):
                a, b = divmod(ind, m)
                df_base = pd.concat([ df_base, DataFrame( \
                    {"p" : [self.pref] * jmult, \
                     "J" : [j] * jmult,\
                     "M" : range(-j,j+1),\
                     "coeff" : [_s(x) for x in vec.flatten()],\
                     "Irrep" : [iname] * jmult,\
                     "mult" : [b+1] * jmult,\
                     "row" : [a+1] * jmult}) ], ignore_index=True)
                     
#            idim = self.dims[ir]
#            iname = self.irrepsname[ir]
#            l = basevecs.shape[0]
#            m = l // idim
#            # build dataframe in whatever basis group was calculated
#            df_base = pd.concat([ df_base, DataFrame(
#                    {"p": [self.pref] * jmult * idim,\
#                     "J" : [j] * jmult * idim,\
#                     "M" : range(-j,j+1) * idim,\
#                     "coeff" : [_s(x) for x in basevecs.flatten()],\
#                     "Irrep" : [iname] * jmult * idim,\
#                     "mult" : [x%m+1 for x in range(l)]*idim,\
#                     "row" : [x//m+1 for x in range(l)]*idim})
#                        ], ignore_index=True)

            # change basis to cartesian basis if using 3D irrep
        return df_base

if __name__ == "__main__":
    print("for checks execute the test script")

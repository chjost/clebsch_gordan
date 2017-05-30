"""Class for groups and their representatives based on quaternions."""

import os
import numpy as np
import itertools as it
from timeit import default_timer as timer

import utils
import quat
import group_generators_quat as gg
from rotations import mapping

class TOh(object):
    """Internally everything is done in the symmetrized spherical harmonics base
    for T1u: (|11>_+, |10>, |11>_-).
    All momenta given are changed to this basis and then transformed with the
    matrix U3, if given.
    """
    def __init__(self, pref=None, withinversion=True, debug=0, irreps=False,
            U2=None, U3=None, U4=None, empty=False):
        if not withinversion:
            raise RuntimeError("only double cover octahedral group works!")
        self.name = "TO"
        self.pref = None if (pref is None) else (
                np.asarray([-1.j*pref[1], pref[2], -pref[0]]))
        self.pref_cart = pref
        self.withinversion = withinversion
        self.debug = debug

        # basis transformation matrices
        if U2 is None:
            self.U2 = np.identity(2)
        else:
            self.U2 = U2
        if U3 is None:
            self.U3 = np.identity(3)
        else:
            self.U3 = U3
        if U4 is None:
            self.U4 = np.identity(4)
        else:
            self.U4 = U4
        if empty:
            return
        
        # set the elements
        # defines elements, lelements
        # see comment in select_elements
        clockalls = timer()
        clock1s = timer()
        self.select_elements()
        self.order = len(self.elements) # order of the group
        clock1e = timer()
        if debug > 1:
            print("element selection: %.2fs" % (clock1e - clock1s))

        # set up multiplication table
        # defines tmult, tmult_global, faithful
        clock1s = timer()
        self.make_mult_table()
        clock1e = timer()
        if debug > 1:
            print("multiplication table: %.2fs" % (clock1e - clock1s))

        # set up list with inverse elements
        # defines linv, linv_global
        clock1s = timer()
        self.make_inv_list()
        clock1e = timer()
        if debug > 1:
            print("inverse elements: %.2fs" % (clock1e - clock1s))

        # determine conjugacy classes
        clock1s = timer()
        # defines tconjugacy
        self.make_conjugacy_relations()
        # defines nclasses, cdim, crep, lclasses
        self.assign_classes()
        clock1e = timer()
        if debug > 1:
            print("conjugacy classes: %.2fs" % (clock1e - clock1s))

        # prepare storage for representations, number of irreps and
        # classes must be the same
        clock1s = timer()
        if irreps:
            self.find_irreps()
        else:
            self.tchar = np.zeros((self.nclasses, self.nclasses), dtype=complex)
        clock1e = timer()
        clockalle = timer()
        if debug > 1:
            print("irrep selection: %.2fs" % (clock1e - clock1s))
            print("total time: %.2fs" % (clockalle - clockalls))

    @classmethod
    def read(cls, path=None, fname=None, p2=0):
        if path is None:
            _path = "./groups"
        else:
            _path = path
        if fname is None:
            _fname = "group_%d.npz" % p2
        else:
            _fname = fname
        filepath = os.path.join(_path, _fname)

        # read file
        fh = np.load(filepath)
        params = fh["params"]
        tmult = fh["tmult"]
        tmultg = fh["tmultg"]
        del fh
        # extract parameters
        pref = params[0]
        debug = params[1]
        withinv = params[2]
        U2 = params[3]
        U3 = params[4]
        U4 = params[5]
        el = params[6]
        lel = params[7]
        flip = params[8]
        flipi = params[9]
        pos = params[10]

        # create new class
        _new = cls(pref=pref, debug=debug, withinversion=withinv,
                   U2=U2, U3=U3, U4=U4, empty=True)
        # set parameters not yet set
        _new.elements = el
        _new.lelements = lel
        _new.order = len(el)
        _new.p2 = 0 if pref is None else np.dot(pref, pref)
        _new.tmult = tmult
        _new.tmult_global = tmultg
        # run the necessary functions
        _new.make_inv_list()
        _new.make_conjugacy_relations()
        _new.assign_classes()
        # create irreps, if necessary
        _new.pos = pos
        if flipi is not None:
            _new.flip_i = flipi.copy()
        if flip is not None:
            _new.flip = flip.copy()
            _new.restore_irreps()

        return _new

    def save(self, path=None, fname=None):
        if path is None:
            _path = "./groups"
        else:
            _path = path
        if fname is None:
            p2 = 0 if self.pref_cart is None else (
                    np.dot(self.pref_cart, self.pref_cart))
            _fname = "group_%d.npz" % p2
        else:
            _fname = fname
        filepath = os.path.join(_path, _fname)
        utils.ensure_write(filepath)

        params = []
        # save pref
        # save debug and withinversion flags
        # save U matrices
        # save lelements
        # save flip vectors
        params.append(self.pref_cart)
        params.append(self.debug)
        params.append(self.withinversion)
        params.append(self.U2)
        params.append(self.U3)
        params.append(self.U4)
        params.append(self.elements)
        params.append(self.lelements)
        try:
            params.append(self.flip)
        except:
            params.append(None)
        try:
            params.append(self.flip_i)
        except:
            params.append(None)
        try:
            params.append(self.pos)
        except:
            params.append(None)

        params = np.asarray(params, dtype=object)

        np.savez(filepath, params=params, tmult=self.tmult,
                 tmultg=self.tmult_global)

    def select_elements(self):
        # self.elements contains the quaternions
        # self.lelements contains the "global" (unique) index of the element,
        # making the elements comparable between different groups
        self.elements = []
        self.lelements = []
        # all possible elements for the double cover octahedral group

        # all elements of O
        for i, el in enumerate(quat.qPar):
            self.elements.append(quat.QNew.create_from_vector(el, 1))
            self.lelements.append(i);
        # all elements of O with inversion
        if self.withinversion:
            for i, el in enumerate(quat.qPar):
                self.elements.append(quat.QNew.create_from_vector(el, -1))
                self.lelements.append(i+24);
        # all elements in the double cover of O
        for i, el in enumerate(quat.qPar):
            self.elements.append(quat.QNew.create_from_vector(-el, 1))
            self.lelements.append(i+48);
        # all elements in the double cover of O with inversion
        if self.withinversion:
            for i, el in enumerate(quat.qPar):
                self.elements.append(quat.QNew.create_from_vector(-el, -1))
                self.lelements.append(i+72);

        if self.pref is None:
            self.p2 = 0
        else:
            self.p2 = np.vdot(self.pref_cart, self.pref_cart)
        # select elements when pref is given
        if self.pref is not None and self.p2 > 1e-6:
            selected = []
            elem = []
            # change reference momentum to T1u basis
            bpref = self.U3.dot(self.pref)
            T1irrep = gg.genT1CMF(self.elements, inv=True, U=self.U3)

            # Go through all elements of T1 and check whether they leave pref 
            # unchanged
            for mat, el, num in zip(T1irrep, self.elements, self.lelements):
                tmp = mat.dot(bpref)
                c1 = utils._eq(tmp - bpref)
                # if mat * bpref == bpref, the quaternion belongs to subgroup 
                # invariant under pref and is appended
                if c1: 
                    selected.append(num)
                    elem.append(el)
            if self.debug > 0:
                print("The group with P_ref = %r has %d elements:" % (
                        self.pref_cart.__str__(), len(elem)))
                tmpstr = ["%d" % x for x in selected]
                tmpstr = ", ".join(tmpstr)
                print("[%s]" % tmpstr)
            # replace self.elements with only the relevant subgroup
            self.elements = elem
            self.lelements = selected

    # calculate all possible multiplications and save the index of the 
    # resulting element
    def make_mult_table(self):
        self.tmult = np.zeros((self.order, self.order), dtype=int)
        self.tmult_global = np.zeros_like(self.tmult)
        for i in range(self.order):
            for j in range(self.order):
                res = -999
                tmp = self.elements[i] * self.elements[j]
                for k in range(self.order):
                    if tmp == self.elements[k]:
                        res = k
                        break
                if res < 0:
                    print("no element found corresponding to %d*%d" % (i,j))
                self.tmult[i,j] = res
                self.tmult_global[i,j] = self.lelements[res]
        self.is_faithful()

    # Check if representation is faithful
    def is_faithful(self):
        # check if complete
        self.faithful = True
        checksum = np.ones((self.order,))*np.sum(self.lelements)
        u, ind = np.unique(self.tmult, return_index=True)
        if len(u) != self.order:
            self.faithful = False
        tmp = np.sum(self.tmult_global, axis=0)
        if not utils._eq(tmp, checksum):
            self.faithful = False
        tmp = np.sum(self.tmult_global, axis=1)
        if not utils._eq(tmp, checksum):
            self.faithful = False

    # calculate all products of elements until the result is one save the 
    # index of the right factor as the inverse of the left.
    def make_inv_list(self):
        self.linv = np.zeros((self.order,), dtype=int)
        self.linv_global = np.zeros((self.order,), dtype=int)
        for i in range(self.order):
            for k in range(self.order):
                tmp = self.elements[i] * self.elements[k]
                if tmp == self.elements[0]:
                    self.linv[i] = k
                    self.linv_global[i] = self.lelements[k]
                    break

    # calculate group conjugates of every element. For every i, j check 
    # whether k^{-1}*j*k = i
    def make_conjugacy_relations(self):
        self.tconjugacy = np.zeros_like(self.tmult, dtype=int)
        # 2 nested for loops
        for i, j in it.product(range(self.order), repeat=2):
            for k in range(self.order):
                k_inv = self.linv[k]
                j_k = self.tmult[j,k]
                k_inv_j_k = self.tmult[k_inv, j_k]
                if k_inv_j_k == i:
                    self.tconjugacy[i,j] = 1
                    break

    # Assign conjugacy classes
    def assign_classes(self):
        # assign a class representative for each class
        tmp = np.ones((self.order,), dtype=int) * -1
        for i, j in it.product(range(self.order), repeat=2):
            if self.tconjugacy[i,j] == 1:
                if tmp[j] == -1:
                    tmp[j] = i
        tmps = np.sort(tmp)
        # Number of classes
        self.nclasses = len(np.unique(tmp))
        # dimension of each class
        self.cdim = np.zeros((self.nclasses,), dtype=int)
        # representative of each class
        self.crep = np.zeros((self.nclasses,), dtype=int)
        icls, j = -1, -1
        for i in range(self.order):
            if tmps[i] != j:
                icls += 1
                j = tmps[i]
                self.cdim[icls] = 1
                self.crep[icls] = j
            else:
                self.cdim[icls] += 1
        cdimmax = np.max(self.cdim)
        # sort all elements into the classes
        # array with index of all elements for all classes and -1 as placeholder
        self.lclasses = np.ones((self.nclasses, cdimmax), dtype=int) * -1
        for i in range(self.nclasses):
            k = -1
            for j in range(self.order):
                if tmp[j] == self.crep[i]:
                    k += 1
                    self.lclasses[i][k] = j

    def check_ortho(self, irrep):
        if len(self.irreps) == 0:
            return True
        char = irrep.characters(self.crep)
        n = len(self.irreps)
        # check row orthogonality
        check1 = np.zeros((n,), dtype=complex)
        for i in range(n):
            res = 0.
            for k in range(self.nclasses):
                res += self.tchar[i,k] * char[k].conj() * self.cdim[k]
            check1[i] = res
        t1 = utils._eq(check1)
        return t1

    def check_orthogonalities(self):
        # check row orthogonality
        check1 = np.zeros((self.nclasses, self.nclasses), dtype=complex)
        for i in range(self.nclasses):
            for j in range(self.nclasses):
                res = 0.
                for k in range(self.nclasses):
                    res += self.tchar[i,k] * self.tchar[j,k].conj() * self.cdim[k]
                check1[i,j] = res
        tmp = self.order * np.identity(self.nclasses)
        t1 = utils._eq(check1, tmp)
        # check column orthogonality
        check2 = np.zeros((self.nclasses, self.nclasses), dtype=complex)
        for i in range(self.nclasses):
            for j in range(self.nclasses):
                res = 0.
                for k in range(self.nclasses):
                    res += self.tchar[k,i] * self.tchar[k,j].conj()
                check2[i,j] = res
        tmp = np.diag(np.ones((self.nclasses,))*self.order/np.asarray(self.cdim))
        t2 = utils._eq(check2, tmp)
        return t1, t2

    def characters_of_SU2(self, j, useinv=True):
        """Character of SU(2) with multiplicity [j] = 2*j+1 for
        all conjugacy classes.

        Parameter
        ---------
        j : int
            The multiplicity of the angular momentum.

        Returns
        -------
        char : ndarray
            The characters of SU(2).
        """
        omegas = np.asarray([self.elements[x].omega() for x in self.crep])
        _inv = np.asarray([float(self.elements[x].i) for x in self.crep])
        _char = np.zeros_like(omegas)
        if j%2 == 0: # half-integer spin
            n = j//2
            for k in range(1, n+1):
                _char += np.cos(0.5 * (2*k-1) * omegas)
        else:
            n = (j-1)//2
            _char += 0.5
            for k in range(1, n+1):
                _char += np.cos(k * omegas)
        _char *= 2.
        if useinv:
            _char *= _inv
        return _char

    def subduction_SU2(self, j, useinv=True):
        """Subduction of continuum SU(2) to lattice irreps
        with multiplity [j] = 2*j+1.

        Parameter
        ---------
        j : int
            The multiplicity of the angular momentum.
        useinv : bool
            Use the inversion flag of the representations.

        Returns
        -------
        irreps : list
            The list of contributing irreps
        """
        multi = self.multiplicity_of_SU2(j, useinv=useinv)
        irreps = [x for x,y in zip(self.irrepsname, multi) if y>0]
        return irreps

    def multiplicity_of_SU2(self, j, useinv=True):
        """Multiplicites of irreps for SU(2) with multiplicity [j] = 2*j+1.

        Parameter
        ---------
        j : int
            The multiplicity of the angular momentum.
        useinv : bool
            Use the inversion flag of the representations.

        Returns
        -------
        multi : ndarray
            The multiplicities of the irreps for the SU(2) representation.
        """
        _char = self.characters_of_SU2(j, useinv=useinv)
        multi = self.tchar.dot(_char*self.cdim)
        multi = np.real_if_close(np.rint(multi/float(self.order)))
        return multi

    def print_mult_table(self, width=24):
        print("multiplication table\n")
        if width > self.order:
            width = self.order
        n = int(self.order)/width
        line = "_".center(width*5, "_")
        for n1 in range(n):
            head = ["%2d" % (x+n1*width) for x in range(width)]
            head = " ".join(head)
            head = "".join(["\n   [", head, "]"])
            for n2 in range(n):
                print(head)
                print(line)
                for i in range(width):
                    ind1 = i+n2*width
                    ind2 = slice(n1*width, (n1+1)*width)
                    tmpstr = ["%2d" % x for x in self.tmult[ind1,ind2]]
                    tmpstr = " ".join(tmpstr)
                    tmpstr = "".join(["%2d | [" % (ind1), tmpstr, "]"])
                    print(tmpstr)

    def print_char_table(self):
        def _tostring(x):
            if np.isclose(x.imag, 0.):
                return " %6.3f " % x.real
            elif np.isclose(x.real, 0.):
                return " %6.3fJ" % x.imag
            else:
                return " %6.3f+" % x.real
        print("")
        tmpstr = [" %6d " % x for x in range(self.nclasses)]
        tmpstr = "|".join(tmpstr)
        tmpstr = "".join(["     |", tmpstr])
        print(tmpstr)
        print("_".center(self.nclasses*9+3, "_"))
        for i in range(self.nclasses):
            tmpstr = [_tostring(x) for x in self.tchar[i]]
            tmpstr = "|".join(tmpstr)
            try:
                tmpstr = "".join([" %4s|" % self.irreps[i].name, tmpstr])
            except IndexError:
                tmpstr = "".join([" %4d|" % i, tmpstr])
            print(tmpstr)

    def print_class_members(self):
        for i in range(self.nclasses):
            tmpstr = ["%d" % x for x in self.lclasses[i] if x != -1]
            tmpstr = ", ".join(tmpstr)
            print("class %2d: %s" % (i, tmpstr))

    def restore_irreps(self):
        self.irreps = []
        self.irrepsname = []
        self.irrepdim = np.zeros((self.nclasses,), dtype=int)
        self.tchar = np.zeros((self.nclasses, self.nclasses), dtype=complex)
        # create the suffix for each flip vector
        self.suffixes = []
        if self.withinversion:
            self.suffixes.append("1u")
            char = "g"
            # one less, since we already appended one
            for i, _ in enumerate(self.flip[:-1]):
                self.suffixes.append("%d%s" % (i//2+2, char))
                if char == "g":
                    char = "u"
                else:
                    char = "g"
        else:
            for i, _ in enumerate(self.flip):
                self.suffixes.append("%d" % (i+1))
        for d in range(1, 5):
            self.get_irreps(d)
            alldone, alldimdone = self.check_possible_dims(d)
            if alldone:
                break
            if alldimdone:
                continue
            if d == 1:
                # create the suffix for each imaginary flip vector
                self.suffixes_i = []
                if self.withinversion:
                    char = "g"
                    for i, _ in enumerate(self.flip_i):
                        self.suffixes_i.append("%d%s" % (i//2+1, char))
                        if char == "g":
                            char = "u"
                        else:
                            char = "g"
                else:
                    for i, _ in enumerate(self.flip_i):
                        self.suffixes_i.append("%d" % (i+1))
                self.get_irreps_imaginary()
            else:
                self.get_irreps(d, special=True)
            alldone = self.check_possible_dims()
            if alldone:
                break

    # schlimmste Funktion die Christian jemals geschrieben hat und hoffentlich je schreiben wird.
    def find_irreps(self):
        self.irreps = []
        self.irrepsname = []
        self.irrepdim = np.zeros((self.nclasses,), dtype=int)
        # character table
        self.tchar = np.zeros((self.nclasses, self.nclasses), dtype=complex)
        # find the possible combinations of irrep dimensions
        # sum_{C} d_C^2 = |G|
        self.find_possible_dims()
        # get flip vectors
        self.find_flip_vectors()
        alldone = self.check_possible_dims()
        # find out brute force which combination of irrep dimensions is realized
        # Loop over irrep dimension from d = 1 to 4 
        # 1. Try generating irreps from SU(2) algebra in d dimensions
        # 2. Try special irrep (not from an algebra) in d dimensions
        # 3. Try to find imaginary flip vectors. For |G| > 10 very expensive -> last resort
        # 4. Break if all irreps are found
        if not alldone:
            for d in range(1, 5):
                # find irreps with simple generators
                self.get_irreps(d)
                alldone, alldimdone = self.check_possible_dims(d)
                if self.debug > 0:
                    print("dim %d: all done: %r, dim done %r" % (d, alldone, alldimdone))
                if alldone:
                    break
                # protect the 1D irreps with imaginary characters,
                # computation is slow with big groups
                if alldimdone or (d == 1 and self.order > 10):
                    continue
                if d == 1:
                    self.find_flip_vectors_imaginary()
                    self.get_irreps_imaginary()
                else:
                    # try groups that can not be generated from an algebra (E, H)
                    self.get_irreps(d, special=True)
                alldone = self.check_possible_dims()
                if alldone:
                    break
        # try irreps with imaginary characters, if group is big
        if not alldone and self.order > 10:
            self.find_flip_vectors_imaginary()
            self.get_irreps_imaginary()
            alldone = self.check_possible_dims()
        #self.print_char_table()
        if not alldone:
            msg = "did not find all irreps, found %d/%d" % (len(self.irreps), self.nclasses)
            print(msg)
            #raise RuntimeError(msg)
        else:
            check1, check2 = self.check_orthogonalities()
            if not check1 or not check2:
                print("row orthogonality: %r\ncolumn orthogonality: %r" % (
                        check1, check2))

    def find_possible_dims(self):
        def _op(data):
            # check if the sum of squares is equal to the order
            check1 = (np.sum(np.square(data)) == self.order)
            return check1
        nmax = int(np.floor(np.sqrt(self.order)))
        # generator for all combinations of dimensions, sorted
        gen = it.combinations_with_replacement(range(1,nmax+1), self.nclasses)
        # filter all combinations which satisfy function
        # _op defined above
        tmp = np.asarray(list(it.ifilter(_op, gen)), dtype=int)
        # generate array with frequency of dimension for every possible
        # combination of dimensions
        self.pos = np.zeros((len(tmp), nmax), dtype=int)
        for n in range(nmax):
            self.pos[:,n] = np.sum(tmp == n+1, axis=1)

    def check_possible_dims(self, dim=None):
        """Return true if all irreps were found.
        If dim is given returns 2 bools, first if all were found,
        second if all for that dim where found.
        """
        # if none found yet
        if len(self.irrepsname) == 0:
            if dim is not None:
                return False, False
            return False
        # if all were found
        if len(self.irreps) == self.nclasses:
            if dim is not None:
                return True, True
            return True
        # maximal dimension of irreps
        nmax = self.pos.shape[1]
        # count the occurences of each dimension
        fre = np.zeros((nmax,), dtype=int)
        for n in range(nmax):
            fre[n] = np.sum(self.irrepdim == n+1)
        # if correct number of irreps already known
        if self.pos.shape[0] == 1:
            if np.any(fre > self.pos[0]):
                if self.debug > 0:
                    tmp = ", ".join(["%dD:%d" % (i+1,n) for i,n in enumerate(self.pos[0])])
                    print("expected: %s" % tmp)
                    tmp = ", ".join(["%dD:%d" % (i+1,n) for i,n in enumerate(fre)])
                    print("found: %s" % tmp)
                raise RuntimeError("found higher number of irreps than expected")
            if dim is not None and dim > 0 and dim < nmax:
                return False, fre[dim-1] == self.pos[0,dim-1]
            if dim is not None:
                return False, True
            # still missing some irreps, otherwise check 2 at
            # the beginning would have fired
            return False
        # keep vector that are still possible
        tmp = []
        for p in self.pos:
            if np.any(fre > p):
                continue
            tmp.append(p)
        self.pos = np.asarray(tmp, dtype=int)
        if self.debug > 0:
            print("possible dims:")
            print(self.pos)
        # if none left something went wrong
        if self.pos.size == 0:
            raise RuntimeError("no possible dimensions found")
        if dim is not None and dim > 0:
            return False, np.all(self.pos[:,dim-1] == fre[dim-1])
        return False

    def append_irrep(self, irrep):
        self.irreps.append(irrep)
        self.irrepsname.append(irrep.name)
        ind = len(self.irreps)-1
        self.irrepdim[ind] = irrep.dim
        irrep.irid = ind
        self.tchar[ind] = irrep.characters(self.crep)

    # sort 1d irreps such that irreps connected by inversion come next to each other
    def sort_flip_vectors(self, flips, special=False):
        """special flag is when flips do not contain the inversion.
        """
        _suffixes = ["%d" % n for n,i in enumerate(flips)]
        _flips = np.asarray(flips)
        if _flips.size == 0:
            raise RuntimeError("no flips to process")
        if self.withinversion is False:
            return _flips, _suffixes
        # get the inversion flags for the classes
        inv = np.asarray([self.elements[x].i for x in self.crep], dtype=int)
        if special:
            count = 1
        else:
            count = 2
            # get the flip that is equal to just the inversion
            for i, f in enumerate(_flips):
                if np.allclose(f, inv):
                    _suffixes[i] = "1u"
                    break
        # treat all other flips and pair them using the inversion
        for i in range(len(_suffixes)):
            # if already assigned, continue
            if _suffixes[i].endswith(("g", "u")):
                continue
            # call the found one "g", this is arbitrary
            _suffixes[i] = "%dg" % count
            # find the corresponding "u" vector
            tmpi = inv * _flips[i]
            ind = 0
            for j, f in enumerate(_flips):
                if np.allclose(tmpi, f):
                    ind = j
                    break
            # set the name
            _suffixes[ind] = "%du" % count
            # increase the number for next iteration
            count += 1
        # set the correct names in the correct order
        if special:
            tmp = ["1g", "1u"]
        else:
            tmp = ["1u"]
        for i in range(2,count):
            tmp.append("%dg" % i)
            tmp.append("%du" % i)
        tmpf = []
        # sort the vectors
        try:
            # If the inversion element (global index 24)
            # is part of the group, g and u can be identified.
            # Inversion is in its own class, we need the
            # class number here.
            ind = np.nonzero(self.crep == 24)[0][0]

            if special:
                start = 1
            else:
                # we can set the first element directly, it is 
                # 1u, which corresponds to the inv vector
                tmpf.append(inv)
                start = 2
            for c in range(start, count):
                f1 = _flips[_suffixes.index("%dg" % c)]
                f2 = _flips[_suffixes.index("%du" % c)]
                # if the character of the inversion class is
                # 1, it is the even (g) representation
                if np.absolute(f1[ind]-1.) < 1e-6:
                    tmpf.append(f1)
                    tmpf.append(f2)
                else:
                    tmpf.append(f2)
                    tmpf.append(f1)
        except IndexError:
            for s in tmp:
                tmpf.append(_flips[_suffixes.index(s)])
        return np.asarray(tmpf), tmp

    # Calculate all 1d irreps
    # idea: take A1 and flip all elements of any class and check whether a irrep was found 8-Q
    # Only finds purely real irreps.
    # See find_flip_vectors_imaginary()
    def find_flip_vectors(self):
        irrep = TOh1D(self.elements)
        flips = []
        n = self.nclasses
        mx_backup = irrep.mx.copy()
        # flip classes
        for k in range(1, n):
            #print("flip %d classes" % k)
            for ind in it.combinations(range(1,n), k):
                fvec = np.ones((n,))
                for x in ind:
                    fvec[x] *= -1
                irrep.flip_classes(fvec, self.lclasses)
                check1 = np.sum(irrep.mx)
                check2 = irrep.is_representation(self.tmult, verbose=False)
                irrep.mx = mx_backup.copy()
                if utils._eq(check1) and check2:
                    flips.append(fvec.copy())
                if self.debug > 1:
                    print("fvec: %s" % fvec.__str__())
                    print("sum check %r, irrep check %r" % (check1, check2))
        flips = np.asarray(flips, dtype=int)
        self.flip, self.suffixes = self.sort_flip_vectors(flips)

    def find_flip_vectors_imaginary(self):
        irrep = TOh1D(self.elements)
        flips = []
        def check_index(i1, i2):
            for x in i2:
                try:
                    i1.index(x)
                    return True
                except ValueError:
                    continue
            return False
        def f_vec(n, ind1, ind2, indm):
            fvec = np.ones((n,), dtype=complex)
            for i in ind1:
                fvec[i] *= 1.j
            for i in ind2:
                fvec[i] *= -1.j
            for i in indm:
                fvec[i] *= -1.
            return fvec

        n = self.nclasses
        mx_backup = irrep.mx.copy()
        count = 0
        if self.debug > 0:
            print("find imaginary flips")
            print("number of classes %d" % n)
        # multiply classes with -1, i and -i,
        # always the same number of classes with +i and -i
        # total number of classes flipped
        for kt in range(1, n):
            if self.debug > 1:
                print("flip %d classes" % (kt))
            # half the number of classes with imaginary flip
            for ki in range(1, kt//2+1):
                if self.debug > 1:
                    print("number of imaginary classes: %d" % (2*ki))
                # get indices for classes with +/- i
                for ind1 in it.combinations(range(1, n), ki):
                    for ind2 in it.combinations(range(1,n), ki):
                        # check if some class is in both index arrays
                        # and skip if it is
                        if check_index(ind1, ind2):
                            if self.debug > 1:
                                print("collision for +/-i:")
                                print("+i: %s" % ind1.__str__())
                                print("-i: %s" % ind2.__str__())
                            continue
                        # get indices for classes with -1
                        for indm in it.combinations(range(1,n), kt-2*ki):
                            # check if some class is already taken
                            # and skip if it is
                            if check_index(ind1, indm):
                                if self.debug > 1:
                                    print("collision for +i/-1:")
                                    print("+i: %s" % ind1.__str__())
                                    print("-i: %s" % indm.__str__())
                                continue
                            # check if some class is already taken
                            # and skip if it is
                            if check_index(ind2, indm):
                                if self.debug > 1:
                                    print("collision for -i/-1:")
                                    print("+i: %s" % ind2.__str__())
                                    print("-i: %s" % indm.__str__())
                                continue
                            fvec = f_vec(n, ind1, ind2, indm)
                            irrep.flip_classes(fvec, self.lclasses)
                            check1 = np.sum(irrep.mx)
                            check2 = irrep.is_representation(self.tmult)
                            irrep.mx = mx_backup.copy()
                            if self.debug > 0:
                                print("fvec: %s" % fvec.__str__())
                                print("sum check %r, irrep check %r" % (check1, check2))
                            if utils._eq(check1) and check2:
                                count += 1
                                flips.append(fvec.copy())
                                # hard-coded due to long runtime
                                if count == 4:
                                    return
        flips = np.asarray(flips, dtype=complex)
        self.flip_i, self.suffixes_i = self.sort_flip_vectors(flips, special=True)

    # Build matrix representations for each irrep
    def get_irreps(self, dim, naming=None, special=False):

        # Set consistent naming scheme (not following any particular convention)
        if special:
            if dim == 2:
                rep = lambda x=None: TOh2Dp(self.elements, self.pref_cart, U=self.U2)
                _name = "Ep" if (naming is None) else naming
            else:
                return
        else:
            if dim == 2:
                rep = lambda x=None: TOh2D(self.elements, U=self.U2)
                _name = "E" if (naming is None) else naming
            elif dim == 3:
                rep = lambda x=None: TOh3D(self.elements, U=self.U3)
                _name = "T" if (naming is None) else naming
            elif dim == 4:
                rep = lambda x=None: TOh4D(self.elements, U=self.U4)
                _name = "F" if (naming is None) else naming
            elif dim == 1:
                rep = lambda x=None: TOh1D(self.elements)
                _name = "A" if (naming is None) else naming
                #raise RuntimeError("there is a special routine for 1D irreps")
            else:
                raise RuntimeError("%dD irreps not implemented" % dim)

        if self.debug > 2:
            print("finding %dD irreps" % dim)
        # build first irrep from algebra (A1g, E1/2g, T1g, F1g)
        # rep is a lambda defined above
        ir = rep()
        ir.name = "".join([_name, "1g"])
        check1 = ir.is_representation(self.tmult, verbose=False)
        check2 = self.check_ortho(ir)
        if check1 and check2:
            self.append_irrep(ir)
        if self.debug > 2:
            print("Irrep %s is representation (%r) and orthogonal (%r)" % (
                    ir.name, check1, check2))
        # Loop over all flip vectors, flip first irrep and check whether also an irrep
        for f, s in zip(self.flip, self.suffixes):
            ir = rep()
            ir.flip_classes(f, self.lclasses)
            ir.name = "".join([_name, s])
            check1 = ir.is_representation(self.tmult)
            check2 = self.check_ortho(ir)
            if check1 and check2:
                self.append_irrep(ir)
            if self.debug > 2:
                print("Irrep %s is representation (%r) and orthogonal (%r)" % (
                        ir.name, check1, check2))

    def get_irreps_imaginary(self):
        for f, s in zip(self.flip_i, self.suffixes_i):
            ir = TOh1D(self.elements)
            ir.flip_classes(f, self.lclasses)
            ir.name = "".join(("K", s))
            if ir.is_representation(self.tmult) and self.check_ortho(ir):
                self.append_irrep(ir)

# irrep class
class TOhRep(object):
    def __init__(self, dimension):
        self.dim = dimension
        self.irid = -1
        self.name = " "
        self.mx = None
        self.char = None
        self.rep = None

    def characters(self, representatives):
        if self.mx is None:
            return np.nan
        elif ((self.char is not None) and
              (self.rep is not None) and
              (utils._eq(self.rep, representatives))):
            return self.char
        else:
            char = np.zeros((len(representatives),), dtype=complex)
            for i, r in enumerate(representatives):
                char[i] = np.trace(self.mx[r])
            self.char = char
            self.rep = representatives
            return char

    def is_representation(self, tmult, verbose=False):
        n = self.mx.shape[0]
        for i in range(n):
            mxi = self.mx[i]
            for j in range(n):
                mxj = self.mx[j]
                mxk = self.mx[tmult[i,j]]
                mxij = mxi.dot(mxj)
                if not utils._eq(mxij, mxk):
                    if verbose:
                        print("elements %d * %d not the same as %d:" % (i,j, tmult[i,j]))
                        print("multiplied:")
                        print(mxi)
                        print(mxj)
                        print("result:")
                        print(mxij)
                        print("expected:")
                        print(mxk)
                        #print("elements %d * %d (%r * %r) not the same as %d (%r / %r)" % (i, j, mxi, mxj, tmult[i,j], mxij, mxk))
                    return False
        return True

    def flip_classes(self, vec, classes):
        """multiply elements by contents of vec based on the class
        they are in."""
        if self.char is not None:
            self.char = None
        for v, c in zip(vec, classes):
            for el in c:
                if el == -1:
                    break
                self.mx[el] *= v

# See get_irreps()

class TOh1D(TOhRep):
    def __init__(self, elements, U=None):
        TOhRep.__init__(self, 1)
        self.name = "TOh1D"
        self.mx = gg.genJ0(elements, U=U)

class TOh2D(TOhRep):
    def __init__(self, elements, U=None):
        TOhRep.__init__(self, 2)
        self.name = "TOh2D"
        self.mx = gg.genJ1_2(elements, U=U)

class TOh3D(TOhRep):
    def __init__(self, elements, U=None):
        TOhRep.__init__(self, 3)
        self.name = "TOh3D"
        self.mx = gg.genT1CMF(elements, U=U)
        #self.mx = gg.genJ1(elements)

class TOh4D(TOhRep):
    def __init__(self, elements, U=None):
        TOhRep.__init__(self, 4)
        self.name = "TOh4D"
        self.mx = gg.genF1CMF(elements, U=U)
        #self.mx = gg.genJ3_2(elements)

class TOh2Dp(TOhRep):
    def __init__(self, elements, pref=None, U=None):
        TOhRep.__init__(self, 2)
        self.name = "TOh2Dp"
        p2 = 0 if (pref is None) else np.dot(pref, pref)
        if p2 in [0, 3]:
            self.mx = gg.genEpCMF(elements, U=U)
        elif p2 in [1, 4]:
            self.mx = gg.genEpMF1(elements, U=U)
        else:
            raise RuntimeError("reference momentum not implemented")

class TOh3Dp(TOhRep):
    def __init__(self, elements, U=None):
        TOhRep.__init__(self, 3)
        self.name = "TOh3D"
        self.mx = gg.gen3D(elements, U=U)

if __name__ == "__main__":
    print("for checks execute the test script")


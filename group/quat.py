"""Class for the quaternions with inversion."""

import numpy as np

import utils

# quaternion parameters for the group O from Table 71.1 in:
# Simon L. Altmann, Peter Herzig, "Point-Group Theory Tables", 
# Second Edition (corrected), Wien (2011)  
V12 = np.sqrt(0.5) # sqrt(1/2)
# [[ lambda, Lambda_1, Lambda_2, Lambda_3 ]]
qPar = np.asarray(
       [[ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 1.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ],
        [ 0.5, 0.5, 0.5, 0.5 ],
        [ 0.5,-0.5,-0.5, 0.5 ],
        [ 0.5, 0.5,-0.5,-0.5 ],
        [ 0.5,-0.5, 0.5,-0.5 ],
        [ 0.5,-0.5,-0.5,-0.5 ],
        [ 0.5, 0.5, 0.5,-0.5 ],
        [ 0.5,-0.5, 0.5, 0.5 ],
        [ 0.5, 0.5,-0.5, 0.5 ],
        [ V12, V12, 0.0, 0.0 ],
        [ V12, 0.0, V12, 0.0 ],
        [ V12, 0.0, 0.0, V12 ],
        [ V12,-V12, 0.0, 0.0 ],
        [ V12, 0.0,-V12, 0.0 ],
        [ V12, 0.0, 0.0,-V12 ],
        [ 0.0, V12, V12, 0.0 ],
        [ 0.0,-V12, V12, 0.0 ],
        [ 0.0, V12, 0.0, V12 ],
        [ 0.0, 0.0,-V12,-V12 ],
        [ 0.0, V12, 0.0,-V12 ],
        [ 0.0, 0.0,-V12, V12 ]])

class QNew(object):
    def __init__(self):
        # entries
        self.q = np.zeros((4,))
        # Inversion flag for double cover
        self.i = int(1)
        self.prec = 1e-6

    @classmethod
    def create_from_vector(cls, vector, inversion):
        tmp = cls()
        _vec = np.asarray(vector)
        tmp.q = _vec.copy()
        _inv = int(inversion)
        tmp.i = _inv
        return tmp

    def __eq__(self, other):
        if not isinstance(other, QNew):
            return False
        if np.allclose(self.q, other.q) and self.i == other.i:
            return True
        return False

    def __ne__(self, other):
        if not isinstance(other, QNew):
            return True
        if not np.allclose(self.q, other.q) or self.i != other.i:
            return True
        return False

    def __abs__(self):
        return np.sqrt(np.dot(self.q, self.q))

    def __neg__(self):
        self.q = -self.q
        self.i = -self.i
        return self

    def __mul__(self, other):
        q1 = self.q
        q2 = other.q
        tvec = np.zeros_like(q1)
        tvec[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        tvec[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        tvec[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        tvec[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        tinv = self.i * other.i
        return QNew.create_from_vector(tvec, tinv)

    def conj(self):
        tvec = self.q * np.asarray([1., -1., -1., -1.])
        return QNew.create_from_vector(tvec, self.i)

    def norm(self):
        return np.dot(self.q, self.q)

    def __str__(self):
        return "[%r, %r, %r, %r] with inversion %d" % (self.q[0], self.q[1], self.q[2], self.q[3], self.i)

    def comp(self, vec):
        return np.allclose(self.q, vec)

    # code inspired by the quaternion package of moble
    # https://github.com/moble/quaternion
    # creates SU(3) rotation matrix from quaternion
    def rotation_matrix(self, inv=False):
        n = self.norm()
        if np.abs(n) < self.prec:
            raise ZeroDivisionError("Norm of quaternion is zero.")
        _q = self.q
        if np.abs(1-n) < self.prec:
            res = np.array(
                [[1-2*(_q[2]**2 + _q[3]**2), 2*(_q[1]*_q[2] - _q[3]*_q[0]),
                    2*(_q[1]*_q[3] + _q[2]*_q[0])],
                 [2*(_q[1]*_q[2] + _q[3]*_q[0]), 1-2*(_q[1]**2 + _q[3]**2),
                    2*(_q[2]*_q[3] - _q[1]*_q[0])],
                 [2*(_q[1]*_q[3] - _q[2]*_q[0]), 2*(_q[2]*_q[3] + _q[1]*_q[0]),
                    1-2*(_q[1]**2 + _q[2]**2)]])
        else:
            res = np.array(
                [[1-2*(_q[2]**2 + _q[3]**2)/n, 2*(_q[1]*_q[2] - _q[3]*_q[0])/n,
                    2*(_q[1]*_q[3] + _q[2]*_q[0])/n],
                 [2*(_q[1]*_q[2] + _q[3]*_q[0])/n, 1-2*(_q[1]**2 + _q[3]**2)/n,
                    2*(_q[2]*_q[3] - _q[1]*_q[0])/n],
                 [2*(_q[1]*_q[3] - _q[2]*_q[0])/n, 2*(_q[2]*_q[3] + _q[1]*_q[0])/n,
                    1-2*(_q[1]**2 + _q[2]**2)/n]])
        if inv:
            res *= self.i
        return res

    # Calculate rotation vector from quaternion
    # Warning: not implemented
    def base(self, inv=False):
        pass

    def R(self, j, mp, m):
        """compute transformation matrix element 
         j         j    0   1   2   3
        R   (Q) = R   (Q , Q , Q , Q )
         m'm       m'm
        
                  -j
        in        __
             j   \     j   j
        [R.u]  = /__  u   R   (Q) ,
             m   m'=j  m'  m'm
        
        according to the formula:
                __    ___________________________
         j     \     /(j-m')(j+m'  )(j-m   )(j+m)    j+m-k    j-m'-k   m'-m+k     k
        R    = /__ \/ ( k  )(m'-m+k)(m'-m+k)( k ) (a)     (a*)      (b)      (-b*)
         m'm    k
                    0     3          2     1
        where a := Q - i.Q  ; b := -Q  -i.Q  .
        
        first three arguments to be provided as multiplicities:
        [J] = 2j+1, [M] = 2m+1, [MP] = 2m'+1, these are always integer
        [-3/2] --> -2; [-1] --> -1; [-1/2] --> 0; [0] --> 1; [1/2] --> 2, etc.
        """
        a   = complex( self.q[0], -self.q[3] ) 
        ac  = complex( self.q[0],  self.q[3] ) #   complex conjugate of a
        b   = complex(-self.q[2], -self.q[1] )
        mbc = complex( self.q[2], -self.q[1] ) # - complex conjugate of b
        res = complex( 0.0 )
        j_p_mp = ( j + mp - 2 ) // 2 # j+m'
        j_m_mp = ( j - mp ) // 2     # j-m'
        j_p_m  = ( j + m  - 2 ) // 2 # j+m
        j_m_m  = ( j - m ) // 2      # j-m
        if j_p_mp < 0 or j_m_mp < 0 or j_p_m < 0 or j_m_m < 0:
            return res

        # prepare constant arrays
        mp_m_m = j_p_mp - j_p_m
        n = np.asarray([j_m_mp, j_p_mp, j_m_m, j_p_m])
        kp = np.asarray([0, mp_m_m, mp_m_m, 0])
        _a = np.asarray([a, ac, b, mbc])
        aexp = np.asarray([j_p_m, j_m_mp, mp_m_m, 0])
        # get range for loop
        k_mx = int(j_p_m if (j_p_m < j_m_mp) else j_m_mp)
        k_mn = int(-j_p_mp+j_p_m if (-j_p_mp+j_p_m > 0) else 0)
        for k in range(k_mn, k_mx+1):
            _k = kp + k
            factor = np.sqrt(np.prod(utils.binomial(n, _k))*complex(1.))
            _aexp = aexp + np.asarray([-k, -k, k, k])
            prod = np.prod(np.power(_a, _aexp))
            res += factor * prod
        return res

    # calculates rotation matrix for a given j
    # j:  Spin 
    def R_matrix(self, j):
        multi = int(2*j+1)
        res = np.zeros((multi, multi), dtype=complex)
        # the sorting is important, start at largest m
        # and decrease
        for im in range(multi):
            m = j - im
            for imp in range(multi):
                mp = j - imp
                res[im, imp] = self.R(multi, int(2*m+1), int(2*mp+1))
        if j%2:
            res *= self.i
        return res
    
    # Calculate rotation angle from quaternion
    def omega(self):
        return 2*np.arccos(self.q[0])

if __name__ == "__main__":
    print("for checks execute the test script")

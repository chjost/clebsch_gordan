"""Deprecated"""

"""Representation of discrete rotations for the double cover octahedral group
with parity.
"""
import numpy as np
import scipy.misc
import quat

class RotObj(object):
    """Rotation object"""
    def __init__(self, vector, omega, prec=1e-6):
        self.prec = prec
        self.vector = np.asarray(vector)
        self.omega = omega

        self.norm = np.sqrt(np.sum(np.abs(vector)))
        if self.norm < 1e-10:
            raise RuntimeError("Cannot work with zero vector")
        self.theta = np.arccos(vector[2]/self.norm)
        self.phi = np.arctan2(vector[1], vector[0])
        self.vector_norm = vector / self.norm
        # needed for the matrix U
        self.v = np.sin(self.omega/2.)*np.sin(self.theta)
        self.u = np.cos(self.omega/2.) - 1.j*np.sin(self.omega/2.)*np.cos(self.theta)

    def __repr__(self):
        return "vector: %r, omega: %.3f, (r, theta, phi) = (%.3f, %.3f, %.3f)" % (
                self.vector, self.omega, self.norm, self.theta, self.phi)

    def __str__(self):
        return self.__repr__()

    def get_vector(self):
        return self.vector_norm

    def get_omega(self):
        return self.omega

    def get_angles(self):
        return (self.theta, self.phi)

    def rot_vector(self, vec):
        # implements Rodrigues formula
        par = self.vector_norm*np.dot(vec, self.vector_norm)
        per = np.cross(self.vector_norm, vec)
        si = np.sin(self.omega)
        co = np.cos(self.omega)
        return vec*co + per*si + par*(1.-co)

    def u_element(self, j, m1, m2):
        prefactor = np.exp(-1.j*(m1-m2)*self.phi)
        fac1 = scipy.misc.factorial([j+m1, j-m1, j+m2, j-m2])
        sqfac1 = np.sqrt(np.prod(fac1))
        #print(sqfac1)
        if m1+m2 >= 0:
            res = 0.
            send = min(j-m1, j-m2)
            #for s in range(0, 50):
            for s in range(send+1):
                fac = np.asarray([s, s+m1+m2, j-m1-s, j-m2-s])
                if np.any(fac < 0):
                    continue
                fac = scipy.misc.factorial(fac)
                tmp1 = 1. if s == 0 else np.power(self.v**2 - 1., s)
                tmp2 = 1. if 2*j-m1-m2-2*s == 0 else np.power(self.v, 2*j-m1-m2-2*s)
                res += sqfac1/np.prod(fac)*tmp1*tmp2
            tmp1 = 1. if m1+m2 == 0 else np.power(self.u, m1+m2)
            tmp2 = np.power(-1.j, 2*j-m1-m2)
            res *= tmp1*tmp2
        else:
            res = 0.
            send = min(j+m1, j+m2)
            #for s in range(0, 50):
            for s in range(send+1):
                fac = np.asarray([s, s-m1-m2, j+m1-s, j+m2-s])
                if np.any(fac < 0):
                    continue
                fac = scipy.misc.factorial(fac)
                tmp1 = 1. if s == 0 else np.power(self.v**2-1., s)
                tmp2 = 1. if 2*j+m1+m2-2*s == 0 else np.power(self.v, 2*j+m1+m2-2*s)
                res += sqfac1/np.prod(fac)*tmp1*tmp2
            tmp1 = 1. if m1+m2 == 0 else np.power(self.u.conj(), -m1-m2)
            tmp2 = np.power(-1.j, 2*j+m1+m2)
            res *= tmp1*tmp2
        return prefactor*res

    def u_matrix(self, j):
        mat = np.zeros((2*j+1, 2*j+1), dtype=complex)
        for m1 in range(-j, j+1):
            for m2 in range(-j, j+1):
                mat[m1+j, m2+j] = self.u_element(j, m1, m2)
        return mat

    def to_quaternion(self):
        v = np.asarray([np.cos(self.omega/2), 0., 0., 0.])
        v[1:] = self.vector_norm * np.sin(self.omega/2)
        return quat.QNew.create_from_vector(v, 1)

_all_rotations = [RotObj([1,0,0], 0),
        RotObj([1,0,0],   np.pi  ), RotObj([0,1,0],   np.pi  ), RotObj([0,0,1],   np.pi  ),
        RotObj([1,0,0],  -np.pi  ), RotObj([0,1,0],  -np.pi  ), RotObj([0,0,1],  -np.pi  ),
        RotObj([1,0,0],   np.pi/2), RotObj([0,1,0],   np.pi/2), RotObj([0,0,1],   np.pi/2),
        RotObj([1,0,0],  -np.pi/2), RotObj([0,1,0],  -np.pi/2), RotObj([0,0,1],  -np.pi/2),
        RotObj([1,0,0], 3*np.pi/2), RotObj([0,1,0], 3*np.pi/2), RotObj([0,0,1], 3*np.pi/2),
        RotObj([1,0,0],-3*np.pi/2), RotObj([0,1,0],-3*np.pi/2), RotObj([0,0,1],-3*np.pi/2),
        RotObj([1,1,1], 2*np.pi/3), RotObj([-1,1,1], 2*np.pi/3), RotObj([-1,-1,1], 2*np.pi/3), RotObj([1,-1,1], 2*np.pi/3),
        RotObj([1,1,1],-2*np.pi/3), RotObj([-1,1,1],-2*np.pi/3), RotObj([-1,-1,1],-2*np.pi/3), RotObj([1,-1,1],-2*np.pi/3),
        RotObj([1,1,1], 4*np.pi/3), RotObj([-1,1,1], 4*np.pi/3), RotObj([-1,-1,1], 4*np.pi/3), RotObj([1,-1,1], 4*np.pi/3),
        RotObj([1,1,1],-4*np.pi/3), RotObj([-1,1,1],-4*np.pi/3), RotObj([-1,-1,1],-4*np.pi/3), RotObj([1,-1,1],-4*np.pi/3),
        RotObj([0,1,1],   np.pi), RotObj([0,-1,1],  np.pi), RotObj([1,1,0],   np.pi),
        RotObj([1,-1,0],  np.pi), RotObj([1,0,1],   np.pi), RotObj([-1,0,1],  np.pi),
        RotObj([0,1,1],  -np.pi), RotObj([0,-1,1], -np.pi), RotObj([1,1,0],  -np.pi),
        RotObj([1,-1,0], -np.pi), RotObj([1,0,1],  -np.pi), RotObj([-1,0,1], -np.pi),
        RotObj([1,1,1], 2*np.pi)]

mapping = [0, 1, 2, 3, 49, 50, 51, 12, 13, 14, 15, 16, 17, 63, 64, 65, 60, 61, 62, 4, 10, 5, 11, 8, 6, 9, 7, 56, 54, 57, 55, 52, 58, 53, 59, 69, 23, 18, 67, 20, 70, 21, 71, 66, 19, 68, 22, 48]

if __name__ == "__main__":
    print("for checks execute the test script")

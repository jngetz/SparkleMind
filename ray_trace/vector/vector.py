from math import sqrt, pow, pi
import numpy as np


# the class of a vector, representative or rays of light
# uses numpy to do vector operations and such

class vector:
    # CI: a vector in 3D space which is defined by the magnitude of movement in
    #     the x, y, and z directions.
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.vec = np.array([x,y,z])

    def defVecNumpy(self, nparray):
        self.vec = nparray.reshape(1,3)
        self.x = self.vec[0]
        self.y = self.vec[1]
        self.z = self.vec[2]

    #PRE:  takes an axis of roation as vector u, and an angle of rotation
    #      theta
    #POST: redefines x,y,z according to np_array
    def rotate(self, axis, theta):
        v = self.vec
        u = axis.vec
        theta = np.deg2rad(theta)
        a1 = np.dot(v,np.cos(theta))
        a2 = np.dot(np.cross(u,v),np.sin(theta))
        a3 = np.multiply(u,np.dot(np.dot(u,v),(1-np.cos(theta))))
        # a1, a2, a3 are all np.arrays of vectors in 3d space which are
        # components to rodriguez's rotation formula
        self.vec = a1+a2+a3
        self.x = self.vec[0]
        self.y = self.vec[1]
        self.z = self.vec[2]
    # vector operations

    # returns an integer of the dot product between this vector, <x,y,z>, and
    #  other vector, <a,b,c> which equals ax + by + cz
    def dotProduct(self, other):
        if isinstance(other,np.ndarray):
            answer = np.dot(self.vec,other)
        else: # we assume that other is a vector or a numpy.ndarray
            answer = np.dot(self.vec, other.vec)
        return answer

    # returns a vector object created by the cross product between this vector,
    #  <x,y,z>, and other vector, <a,b,c> which is <yc-bz, za-cx, xb-ay>
    def crossProduct(self, other):
        if isinstance(other,np.ndarray):
            answer = np.cross(self.vec,other)
        else: # we assume that other is a vector or a numpy.ndarray
            answer = np.cross(self.vec, other.vec)
        return vector(answer[0], answer[1], answer[2])

    def magnitude(self):
        return np.linalg.norm(self.vec)

    def unit(self):
        M = self.magnitude()
        return vector(self.x/M, self.y/M, self.z/M)

    def scale(self, scale):
        self.vec = np.multiply(self.vec, scale)
        self.x *= scale
        self.y *= scale
        self.z *= scale

    # Overload summation and subtraction
    def __add__(self, other):
        out = self.vec + other.vec
        return vector(out[0],out[1],out[2])

    def __sub__(self, other):
        out = self.vec - other.vec
        return vector(out[0],out[1],out[2])

    def __str__(self):
        return "<"+str(self.x)+", "+str(self.y)+", "+str(self.z)+">"

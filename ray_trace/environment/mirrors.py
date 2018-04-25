# the mirrors that we can use in the environment
import random as R
from vector import vector as V
import numpy as np

class mirrors:

    #PRE:  T is the type of the mirror to make
    #      S is the size of the mirror to make
    #      O is the surfaces overall orientation vector with 0 z component
    #      L is location as vector to top left corner
    #POST: creates an NxN array of vectors relative to the mirror surface which
    #      represent the normal to the plane.
    def __init__(self, T, S, O=V.vector(0,0,1), L=V.vector(0,0,0)):
        self.seed = 123456
        self.size = S
        self.orientation = V.vector(O[0], O[1], O[2])
        self.top_left = L
        # this is the direction the pixels are moving from "0,0" of the mirror
        # we assume that all elements have 0 z component for their orientation,
        # and as such, [0,0,1] is orthogonal to self.orientation
        out = np.cross(self.orientation.vec, [0,0,1])
        self.right_vec = V.vector(out[0],out[1],out[2])
        #self.right_vec is a vector of the direction of pixels following the
        #right of the top left since cross products follow the right hand rule
        if (T == True):
            self.sheet = self.makeGlitter()
        else:
            self.sheet = self.makeFlat()

    #PRE:  initially assumes the orientation of the sheet to be [0 0 1]
    #      and rotates via the vector rotate method the reflector to the correct
    #      orientation in a 3D environment
    #POST: makes a sheet of glitter oriented to the space
    def makeGlitter(self):
        R.seed(self.seed)
        paper = []
        for i in range(0,self.size):
            row = []
            for j in range(0,self.size):
                mx = float(format(R.uniform(-1,1),'.2'))
                my = float(format(R.uniform(-1,1),'.2'))
                mz = np.absolute(float(format(R.uniform(-1,1),'.2')))
                reflector = V.vector(mx, my, mz)
                # reflector = reflector.unit() # not necessary
                cos_theta = (reflector.dotProduct(self.orientation.vec)/
                            (reflector.magnitude()*np.linalg.norm(self.orientation.vec)))
                theta = np.degrees(np.arccos(cos_theta))
                u = reflector.crossProduct(self.orientation.vec).unit()
                # u is perpendicular to the reflector and the axis of ratation
                reflector.rotate(u,theta)
                row.append(reflector)
            paper.append(row)
        return paper

    def makeFlat(self):
        paper = []
        f = [0,0,1] # surface normal of flat mirror with frame of reference
                    # +z away from overall orientation
        reflector = V.vector(f[0],f[1],f[2])
        cos_theta = (reflector.dotProduct(self.orientation)/(reflector.magnitude()*np.linalg.norm(self.orientation.vec)))
        theta = np.degrees(np.arccos(cos_theta))
        # print("theta: ",theta)
        u = reflector.crossProduct(self.orientation).unit()
        # print("unit axis of rotation: ",str(u))
        # u is perpendicular to the reflector and the axis of rotation
        # print("pre-rotation: ",str(reflector))
        reflector.rotate(u,theta)
        # print("post-rotation: ",str(reflector))
        for i in range(0,self.size):
            row = []
            for j in range(0,self.size):
                row.append(reflector)
            paper.append(row)
        return paper


    # Writing a pixel by pixel color representation of the mirror surface.
    # R, G, -1 and 1, -1 is 0, 1 is 300, 0 is 150
    # B, 0 and 1, 0 is 0, 1 is 300
    # writing as a .ppm file
    def draw_img(self):
        # proper .ppm header
        print_str = "P3\n"+str(self.size)+" "+str(self.size)+"\n300\n"

        for i in range(0,self.size):
            for j in range(0,self.size):
                r = int(np.ceil((np.absolute(self.sheet[i][j].x))*300))
                g = int(np.ceil((np.absolute(self.sheet[i][j].y))*300))
                b = int(np.ceil((np.absolute(self.sheet[i][j].z))*300))
                color = str(r)+" "+str(g)+" "+str(b)+"\n"
                # print(color)
                print_str += color
        img = open("mirror2.ppm","r+")
        # print(print_str)
        img.write(print_str)

    def __str__(self):
        print_str = ""
        for i in range(0,self.size):
            for j in range(0,self.size):
                print_str += str(self.sheet[i][j])+"\t"
            print_str += "\n"
        return print_str

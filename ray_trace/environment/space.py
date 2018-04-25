# class of 3D space which returns a rendered image of the space from a camera
# location given on the "render" method.
# can place a irregular mirro-reflector surface with a specific random-seed into
#  space
# can place an image into the space
import mirrors as M
import images as I
import camera as C
from vector import vector as V
import numpy as np
from PIL import Image as img

class space:
    #PRE:  can take a list of Mirror or picture objects to place into space
    #POST: sets self.origin to vector <0,0,0>
    #      sets the self.objects array equal to whatever was passed in.
    def __init__(self, objects=[]):
        self.origin  = V.vector(0,0,0)
        self.objects = objects
        self.camera = None
        self.focal_pt = None

    #PRE:  L = top left corner as a vector
    #      M = mirror object
    #POST: space now holds the information of
    def placeGlitterPaper(self, M):
        self.objects.append(M)

    def placeImage(self, I):
        self.objects.append(I)

    #PRE:  L is location as vector to top_left corner of sensor array
    #      F is the focal point as vector pointing to the 3D location of the
    #      point
    #      S is the int size of the sensor array SxS
    def placeCamera(self, O, S, F, L):
        self.camera = C.camera(O,S,F,L)

    #PRE:  expects a mirror, camera, and image to be placed into the space
    #      optionally takes a location to write the image to
    #POST: writes an image into the Reflections folder with the same
    #      name as from Originals of the SxS image of the colors that appear on
    #      the sensor
    def render(self, out_file="./temp.png"):
        # right_dir defines the direction we are taking pixel steps
        # as a unit vector
        right_dir = self.camera.right_vec.unit()
        for i in range(0,self.camera.size): # row
            for j in range(0,self.camera.size): # column
                sensor_pixel = self.camera.top_left.vec
                del_x  = np.dot(i,np.array([0,0,-1]))
                #from top_left x, we move i*[0,0,-1]
                del_xy = np.dot(j,right_dir.vec)
                #from top_left y, we move j*right_dir
                # origin of the ray is np.array
                sensor_pixel = np.add(sensor_pixel, np.add(del_x,del_xy))
                self.camera.sensor[i][j] = self.trace_ray(sensor_pixel,self.camera.focal.vec)
        image_write = img.fromarray(self.camera.sensor.astype('uint8'))
        # image_write = image_write.convert("L")
        image_write.save(out_file)

    #PRE:  p1 is the origin of the ray to trace, and p2 is the other defining
    #      point of the ray. both are numpy arrays of shape 1,3
    #POST: returns the color found from tracing the ray
    def trace_ray(self, p1, p2):
        # inc is the vector of the ray
        out = p2 - p1
        inc = V.vector(out[0],out[1],out[2])
        color = 0 # in PIL, 0 is black and 255 is white
        check = self.check_intersect(inc, p1)
        # check has object in
        if (isinstance(check[0],M.mirrors)): # the ray intersected with a mirror
            # reflection of v over a vector n is given by
            # -1*((2*np.dot(v,n)*n)-v). s_norm is n, and inc is v
            s_norm = check[2] #surface normal of the point on the mirror it hits
            out = self.reflect(inc,s_norm)
            p1_1 = check[1]+(np.dot(out.unit().vec, 2))
            # location of intersection with mirror plus a little distance
            p2_1 = p1_1+(np.dot(out.unit().vec, 2))
            # print("before trace_ray")
            # print("p1_1",p1_1,"p2_1",p2_1)
            color = self.trace_ray(p1_1,p2_1)
        elif(isinstance(check[0],I.images)):
            color = check[2]
        return color

    #PRE:  ray is a vector
    #      p0 is the start point of the ray as numpy array
    #POST: returns the object, the point at which the ray intersects, and
    #      the item stored at the n,m position which corresponds to the
    #      point of intersection
    #      the object as a 3-array. if there is no intersection then the
    #      object returned is None type
    #      success returns [object, np.array intersection point, [n,m]]
    def check_intersect(self,ray,l0):
        return_val = [None, None, None]
        v = ray.unit()
        for item in self.objects:
            n = item.orientation.unit()
            p0 = item.top_left.vec
            #calculate np.dot(v,n) to determine if parallel
            if np.dot(v.vec,n.vec) != 0: #not parallel
                # print("not parallel")
                # distance to plane it intersects
                d = np.divide(np.dot((p0-l0),n.vec),np.dot(v.vec,n.vec))
                intersect = np.ceil(np.multiply(d,v.vec)+l0)
                # convert point of contact to n,m in array
                # n is row, m is column
                n = p0[2]-intersect[2]
                # n is difference in z from top left to intersect
                m = np.divide(intersect[0] - p0[0], item.right_vec.unit().vec[0])
                # m is the difference of intersect and top left divide by the
                # direction of the rightward pixel vector
                # a np.array of the location
                # of intersection
                #check bounds of current item and intersect
                # print("n",n,"m",m)
                # print("item.size",item.size)
                if (d > 0 and
                   m >= 0 and
                   m < item.size and
                   n >= 0 and
                   n < item.size):
                    # print("in bounds")
                    return_val = [item,intersect,item.sheet[int(n)][int(m)]]
                # otherwise do nothing

        return return_val
    #PRE:  called only if the mirror object is hit. expects inc and norm to be
    #      vector objects
    #POST: returns a vector object congruent to inc reflected across the normal vector
    def reflect(self, inc, norm):
        out = np.add(np.dot(-2,np.dot(inc.dotProduct(norm),norm.vec)),inc.vec)
        return V.vector(out[0], out[1], out[2])

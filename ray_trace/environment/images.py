from vector import vector as V
from PIL import Image as img
import numpy as np

class images:
    #PRE:  imgloc is the location of the image in the directory as string
    #      O is the orientation of the image in space as a vector with 0 z
    #      component
    #      L is the location of the image as a vector to top left corner
    def __init__(self, O, imgloc, L=np.array([0,0,0])):
        # properties
        self.img_loc = imgloc
        image = img.open(imgloc)
        image = image.convert("L")
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0]))
        self.sheet = im_arr
        self.orientation = O
        self.size = image.size[0]
        # this is the direction the pixels are moving from "0,0" of the mirror
        # we assume that all elements have 0 z component for their orientation,
        # and as such, [0,0,1] is orthogonal to self.orientation
        out = np.cross(self.orientation.vec, [0,0,1])
        self.right_vec = V.vector(out[0],out[1],out[2])
        #self.right_vec is a vector of the direction of pixels following the
        #right of the top left since cross products follow the right hand rule
        self.top_left = L

    def __str__(self):
        return np.array_str(self.sheet)

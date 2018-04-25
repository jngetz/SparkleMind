from environment import space as S
from environment import mirrors as M
from environment import images as I
from environment import camera as C
from vector import vector as V
import sys
import numpy as np

input_file = sys.argv[1] # relative path to the file we will read in. will be string.
# expected to exist
output_file = sys.argv[2] # relative path to the file to write when finished
#expected to exist
# print(input_file)
# print(output_file)
# exit()

m1 = M.mirrors(True,28, np.array([0,-1,0]), V.vector(14,0,14)) # presupose facing +y before orientation vector
print("mirror generated")
i1 = I.images(V.vector(-1,1,0),input_file, V.vector(2,-25,14))
env = S.space()
env.placeCamera(V.vector(1,1,0), 28, V.vector(-2,-4,0), V.vector(-22,-5,14))
env.placeGlitterPaper(m1)
env.placeImage(i1)
env.render(output_file)
##
# i1 = I.images(V.vector(0,-1,0), "./Originals/test1.jpeg", V.vector(500,500,500))
# env = S.space()
# env.placeCamera(V.vector(0,1,0), 1000, V.vector(0,0,0), V.vector(-500,-500,500))
# env.placeImage(i1)
# env.render()

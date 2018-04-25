from PIL import Image as img
import sys
# relative paths to images
img1 = sys.argv[1]
img2 = sys.argv[2]

# expect each image to be 1000x1000 takes the sum of the difference of each
# pixel and squares it which is the "error" of the sameness of img2 to img1

i1 = img.open(img1)
i2 = img.open(img2)
sd = 0 # as far as we know the sum difference is 0 currently
for i in range(0,i1.size[0]):
    for j in range(0,i1.size[1]):
        sd += (i2.getpixel((i,j)) - i1.getpixel((i,j)))
print("Sum Square Difference",sd*sd)

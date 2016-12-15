import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
img=mpimg.imread('a.png')
lum_img = img[:,:,0]
# plt.imshow(lum_img)
plt.imshow(lum_img, cmap="hot")
plt.show()

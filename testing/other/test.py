import numpy as np
import matplotlib.pyplot as plt
from cellpose import plot, utils, io

dat = np.load('_seg.npy', allow_pickle=True).item()
img = io.imread('data_from_paper.ome.tif')

# plot image with masks overlaid
mask_RGB = plot.mask_overlay(img, dat['masks'],
                        colors=np.array(dat['colors']))

# plot image with outlines overlaid in red
outlines = utils.outlines_list(dat['masks'])
plt.imshow(img)
for o in outlines:
    plt.plot(o[:,0], o[:,1], color='r')
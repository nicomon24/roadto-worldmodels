'''
    Collection of various utility functions.
'''

import numpy as np
import matplotlib.pyplot as plt

def NCHW(x):
    x = np.array(x)
    if len(x.shape) == 4:
        return np.transpose(x, (0, 3, 1, 2))
    elif len(x.shape) == 3:
        return np.transpose(x, (2, 0, 1))
    else:
        raise Exception("Unrecognized shape.")

def NHWC(x):
    x = np.array(x)
    if len(x.shape) == 4:
        return np.transpose(x, (0, 2, 3, 1))
    elif len(x.shape) == 3:
        return np.transpose(x, (1, 2, 0))
    else:
        raise Exception("Unrecognized shape.")

def imshow_bw_or_rgb(img):
    if img.shape[-1] == 1:
        plt.imshow(img[:,:,0], cmap="Greys")
    elif img.shape[-1] == 3:
        plt.imshow(img)
    else:
        raise Exception('Unrecognized image format')

def side_by_side(img1, img2, SIZE=4):
    if len(img1.shape) == 2:
        return np.concatenate([img1, np.ones((img1.shape[0], SIZE)), img2], axis=1)
    elif len(img1.shape) == 3:
        return np.concatenate([img1, np.ones((img1.shape[0], SIZE, img1.shape[2])), img2], axis=1)
    else:
        raise Exception("Unrecognized observation format!")

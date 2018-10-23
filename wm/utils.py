'''
    Collection of various utility functions.
'''

import numpy as np

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
    import matplotlib.pyplot as plt
    if len(img.shape) == 2:
        plt.imshow(img, cmap="Greys")
    elif len(img.shape) == 3 and img.shape[-1] == 1:
        plt.imshow(img[:,:,0], cmap="Greys")
    elif len(img.shape) == 3 and img.shape[-1] == 3:
        plt.imshow(img)
    else:
        raise Exception('Unrecognized image format')

def side_by_side(img1, img2, SIZE=4):
    '''
        This function
    '''
    import matplotlib.pyplot as plt
    if len(img1.shape) == 2:
        return np.concatenate([img1, np.ones((img1.shape[0], SIZE)), img2], axis=1)
    elif len(img1.shape) == 3:
        if img1.shape[-1] == 1:
            return side_by_side(img1[:,:,0], img2[:,:,0], SIZE)
        return np.concatenate([img1, np.ones((img1.shape[0], SIZE, img1.shape[2])), img2], axis=1)
    else:
        raise Exception("Unrecognized observation format!")

def select_n_workers(n_workers):
    '''
        This function safely selects the number of workers to use,
        even when the scheduler affinity call is not present (i.e. non linux os).
    '''
    # Safely get affinity
    try:
        affinity = len(os.sched_getaffininity(0))
    except:
        affinity = -1
    # Select
    if affinity > 0 and n_workers == -1:
        return affinity # Use all cores
    elif affinity > 0 and n_workers > 0:
        return min(affinity, n_workers) # Use n_workers if possible
    elif affinity == -1 and n_workers > 0:
        return n_workers # Use the provided n_workers
    else:
        return 1 # Safely set only 1 worker

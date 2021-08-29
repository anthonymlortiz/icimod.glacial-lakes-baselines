import os
import numpy as np
import cv2
import time


# General utils functions
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


class Timer():
    '''A wrapper class for printing out what is running and how long it took.
    
    Use as:
    ```
    with utils.Timer("running stuff"):
        # do stuff
    ```

    This will output:
    ```
    Starting 'running stuff'
    # any output from 'running stuff'
    Finished 'running stuff' in 12.45 seconds
    ```
    '''
    def __init__(self, message, verbose=True):
        self.message = message
        self.verbose = verbose

    def __enter__(self):
        self.tic = float(time.time())
        if self.verbose:
            print("Starting '%s'" % (self.message))

    def __exit__(self, type, value, traceback):
        if self.verbose:
            print("Finished '%s' in %0.4f seconds" % (self.message, time.time() - self.tic))


def batch_resize(image_batch, size, interpolation):
    resized_image_batch = np.zeros((image_batch.shape[0], size[1], size[0]))
    for i in range(image_batch.shape[0]):
        img = cv2.resize(image_batch[i], size, interpolation)
        resized_image_batch[i] = img
    return resized_image_batch

import os
import numpy as np
import cv2


## General utils functions
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def batch_resize(image_batch, size, interpolation):
    resized_image_batch = np.zeros((image_batch.shape[0], size[1], size[0]))
    for i in range(image_batch.shape[0]):
        img = cv2.resize(image_batch[i], size, interpolation)
        resized_image_batch[i] = img
    return resized_image_batch







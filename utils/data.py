import numpy as np
import torch
import random
import cv2
import rasterio.mask
import geopandas as gpd
import shapely.geometry as sg
from scipy import ndimage
from torch.utils.data import Dataset


class ToyExperiment(Dataset):

    def __init__(self, x, y, dts, sdts, N=10):
        x = x / x.max()
        self.p = []
        for i in range(y.shape[0]):
            self.p.append(extreme_points(y[i]))

        # concatenate gaussian extreme points with original x
        self.p_blur = gaussian_convolve(x.shape[1:], np.vstack(self.p))\
            .sum(0, keepdims=True)
        self.x = np.vstack((x, self.p_blur))
        self.x = torch.from_numpy(self.x).float()

        self.y = torch.from_numpy(y).sum(axis=0, keepdims=True)
        self.dts = torch.from_numpy(dts).sum(axis=0, keepdims=True)
        self.sdts = torch.from_numpy(sdts).sum(axis=0, keepdims=True)
        self.patch_size = [128, 128]
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, ix):
        _, w, h = self.x.shape
        w_min = random.randint(0, w - self.patch_size[0])
        h_min = random.randint(0, h - self.patch_size[1])
        w_ix = np.arange(w_min, w_min + self.patch_size[0])
        h_ix = np.arange(h_min, h_min + self.patch_size[1])

        x = self.x[:, w_ix[None, :], h_ix[:, None]]
        return x,\
            self.y[:, w_ix[None, :], h_ix[:, None]],\
            self.dts[:, w_ix[None, :], h_ix[:, None]],\
            self.sdts[:, w_ix[None, :], h_ix[:, None]]


def extreme_points(mask, pert=0):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]


    # List of coordinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)+pert)), # left
                     find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x)-pert)), # right
                     find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y)+pert)), # top
                     find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)-pert)) # bottom
                     ])


def make_gaussian(size, center, sigma=10):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(size[1])
    y = np.arange(size[0])[:, np.newaxis]
    x0, y0 = center
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def gaussian_convolve(dim, p, sigma=10):
    """ Make the ground-truth for  landmark.
    dim: The shape of the underlying image
    p: A numpy array containing centers of all the points to draw heatmaps at
    sigma: sigma of the Gaussian.
    """
    h, w = dim
    gt = np.zeros((p.shape[0], h, w))
    for i in range(p.shape[0]):
        gt[i] = make_gaussian((h, w), center=p[i, :], sigma=sigma)
    return gt


def sdt_i(yi, dt_max=30):
    p = cv2.findContours(yi.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    contours = cv2.drawContours(np.zeros(yi.shape), p, -1, 1)
    dt = ndimage.distance_transform_edt(contours == 0)

    sdt = dt.copy()
    sdt[sdt > dt_max] = dt_max
    sdt[yi > 0] *= -1
    return dt, sdt


def sdt(y, dt_max=30):
    dts, sdts = np.zeros(y.shape), np.zeros(y.shape)
    for i, yi in enumerate(y):
        dts[i], sdts[i] = sdt_i(yi, dt_max)
    return dts, sdts


def mask(y, img):
    extent = gpd.GeoDataFrame(
        index=[0],
        crs=y.crs,
        geometry=[sg.box(*img.bounds)]
    )
    y_extent = gpd.overlay(y, extent)

    masks, p = [], []
    for geom in y_extent["geometry"]:
        mask_i, _, _ = rasterio.mask.raster_geometry_mask(img, [geom], invert=True)
        if not np.all(mask_i == 0):
            masks.append(mask_i)
            p.append(extreme_points(mask_i))

    return np.stack(masks), p


def preprocessor(img, y):
    """
    Helper for processing x, y for levelsets

    Inputs
    x: rasterio image object, with bounds
    y: geopandas data.frame used to create training data masks
    """
    x = img.read()
    y, extreme_polys = mask(y, img)
    dist, signed_dist = sdt(y)
    extreme_hm = gaussian_convolve(x.shape[1:], np.vstack(extreme_polys))
    y = [z.sum(0) for z in [y, extreme_hm, dist, signed_dist]]
    return np.nanmean(x, (1, 2)), np.nanstd(x, (1, 2)), np.stack(y)

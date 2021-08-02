import numpy as np
import random
import csv
import rasterio.mask
import geopandas as gpd
import shapely.geometry as sg
import pathlib
import pandas as pd
import shutil
from scipy import ndimage
from tqdm import tqdm


def extreme_points(mask, pert=0):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    # List of coordinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)+pert)),  # left
                     find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x)-pert)),  # right
                     find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y)+pert)),  # top
                     find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)-pert))  # bottom
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


def sdt_i(yi, dist_max=10):
    dt_inner = ndimage.distance_transform_edt(yi.copy() == 0)
    dt_outer = ndimage.distance_transform_edt(yi.copy() == 1)
    sdt = dt_inner - dt_outer
    sdt[sdt > dist_max] = dist_max
    return dt_outer, -sdt


def sdt(y, dist_max=10):
    dts, sdts = np.zeros(y.shape), np.zeros(y.shape)
    for i, yi in enumerate(y):
        dts[i], sdts[i] = sdt_i(yi, dist_max)
    return dts, sdts


def mask(y, img):
    if (len(y)) == 0:
        mask = np.zeros((1, img.meta["height"], img.meta["width"]))
        p = [[0, 0], [0, 1], [1, 0], [1, 1]]
        return mask, p

    extent = gpd.GeoDataFrame(
        index=[0],
        crs=y.crs,
        geometry=[sg.box(*img.bounds)]
    )
    y_extent = gpd.overlay(y, extent)

    # build the masks, one polygon at a time
    masks, p = [], []
    for geom in y_extent["geometry"]:
        mask_i, _, _ = rasterio.mask.raster_geometry_mask(img, [geom], invert=True)
        if not np.all(mask_i == 0):
            masks.append(mask_i)
            p.append(extreme_points(mask_i))

    # crop to the same shape as img, if slightly off
    masks = np.stack(masks)
    if masks.shape[1] != img.meta["height"] or masks.shape[2] != img.meta["width"]:
        masks = masks[:, :img.meta["height"], :img.meta["width"]]

    return masks, p


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
    maxes = [z.max(0) for z in [y, extreme_hm, dist, signed_dist]]
    y, meta = maxes[0][np.newaxis, ...], np.stack(maxes[1:])
    return np.nanmean(x, (1, 2)), np.nanstd(x, (1, 2)), y, meta


def save_raster(z, meta, transform, path, exist_ok=True):
    meta.update({
        "driver": "GTiff",
        "height": z.shape[1],
        "width": z.shape[2],
        "count": z.shape[0],
        "transform": transform,
        "dtype": rasterio.float32
    })

    path.parent.mkdir(parents=True, exist_ok=exist_ok)
    with rasterio.open(path, "w", **meta) as f:
        f.write(z.astype(np.float32))


def inference_paths(x_dir, meta_dir, infer_dir, subset_size=None):
    fn = list(pathlib.Path(x_dir).glob("*tif"))
    meta_fn = list(pathlib.Path(meta_dir).glob("*tif"))
    fn.sort(), meta_fn.sort()

    out_fn_y = [infer_dir / (f.stem + "-pred.tif") for f in fn]
    out_fn_prob = [infer_dir / (f.stem + "-prob.tif") for f in fn]

    result = pd.DataFrame({
        "sample_id": [f.stem for f in fn],
        "GLID": [f.stem.split()[0] for f in fn],
        "fn": fn,
        "meta_fn": meta_fn,
        "out_fn_y": out_fn_y,
        "out_fn_prob": out_fn_prob
    }).set_index("sample_id")

    if subset_size is not None:
        result = result[:subset_size]
    return result


def eval_paths(infer_dir):
    fn = list(pathlib.Path(infer_dir).glob("*-pred.tif"))
    return pd.DataFrame({
        "path": fn,
        "sample_id": [str(f.stem).replace("-pred", "") for f in fn]
    })

def preprocess_dir(in_dir, y):
    if (in_dir / "images").exists():
        shutil.rmtree(in_dir / "images")
        shutil.rmtree(in_dir / "labels")
    (in_dir / "images").mkdir(parents=True)
    (in_dir / "meta").mkdir(parents=True)

    scene_list = list(in_dir.glob("*.tif"))
    fields = ["scene"] + sum([[f"{s}_{i}" for i in range(11)] for s in ["mean", "sdev"]], [])
    f = open(in_dir / "statistics.csv", "a")
    writer = csv.writer(f)
    writer.writerow(fields)

    print(f"preprocessing {in_dir}")
    for scene in tqdm(scene_list):
        img = rasterio.open(scene)
        mean, std, label, meta = preprocessor(img, y)
        save_raster(label, img.meta, img.transform, in_dir / f"labels/{scene.stem}-labels.tif")
        save_raster(meta, img.meta, img.transform, in_dir / f"meta/{scene.stem}-meta.tif")
        writer.writerow([str(scene.stem)] + list(np.hstack([mean, std])))

    f.close()
    [shutil.move(str(s), in_dir / "images") for s in scene_list]

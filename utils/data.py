import numpy as np
import random
import csv
import rasterio.mask
import geopandas as gpd
import shapely.geometry as sg
import pathlib
import pandas as pd
import shutil
from scipy import ndimage as ndi
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed
from shapely.ops import transform
import pyproj
from warnings import warn, filterwarnings
filterwarnings("ignore", category=FutureWarning)


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


def make_gaussian(size, center, sigma=20):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(size[1])
    y = np.arange(size[0])[:, np.newaxis]
    x0, y0 = center
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def inverse_gaussian_gradient(image, alpha=100.0, sigma=5.0):
    """Inverse of gradient magnitude.
    Compute the gaussian of magnitude of the gradients in the image and then inverts the
    result in the range [0, 1]. Flat areas are assigned values close to 1,
    while areas close to borders are assigned values close to 0.
    """
    gradnorm = ndi.gaussian_gradient_magnitude(image, sigma, mode='nearest')
    return 1.0 / np.sqrt(1.0 + alpha * gradnorm)


def gaussian_convolve(dim, p, sigma=20):
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


def sdt_i(yi, dist_max=40):
    dt_inner = ndi.distance_transform_edt(yi.copy() == 0)
    dt_outer = ndi.distance_transform_edt(yi.copy() == 1)
    dt = dt_inner + dt_outer
    sdt = dt_inner - dt_outer

    # truncate at dist_max
    dt[dt > dist_max] = dist_max
    sdt[sdt > dist_max] = dist_max
    sdt[sdt < -dist_max] = -dist_max
    return dt, sdt


def sdt(y, dist_max=40):
    dts, sdts = np.zeros(y.shape), np.zeros(y.shape)
    for i, yi in enumerate(y):
        dts[i], sdts[i] = sdt_i(yi, dist_max)
    return dts, sdts


def buffer_polygon_in_meters(polygon, buffer, percentage=0.4):
    proj_meters = pyproj.Proj('epsg:3857')
    proj_latlng = pyproj.Proj('epsg:4326')

    project_to_meters = partial(pyproj.transform, proj_latlng, proj_meters)
    project_to_latlng = partial(pyproj.transform, proj_meters, proj_latlng)
    pt_meters = transform(project_to_meters, polygon.unary_union)

    buffer_meters = pt_meters
    while buffer_meters.area > percentage * pt_meters.area:
        buffer_meters = buffer_meters.buffer(buffer)

    buffer_polygon = transform(project_to_latlng, buffer_meters)
    return gpd.GeoDataFrame(geometry=[buffer_polygon], crs=polygon.crs)


def get_buffer_from_area(area, step_percentage=-1):
    #Area in sq km
    prop = (100 + step_percentage) / 100.0
    sign = 1 if step_percentage > 0 else -1
    return sign * (1 - np.sqrt(prop)) * np.sqrt(1000000 * area)


def reverse_buffer(y, img, percentage=-25):
    buffer_size = get_buffer_from_area(y["Area"].values[0])
    y_init = buffer_polygon_in_meters(y, buffer_size)
    return mask(y_init, img)[0]


def mask(y, img):
    extent = gpd.GeoDataFrame(index=[0], crs=y.crs, geometry=[sg.box(*img.bounds)])
    y_extent = gpd.overlay(y, extent)

    # build the masks, one polygon at a time
    masks, p = [], []
    for geom in y_extent["geometry"]:
        mask_i, _, _ = rasterio.mask.raster_geometry_mask(img, [geom], invert=True)
        if not np.all(mask_i == 0):
            masks.append(mask_i)
            p.append(extreme_points(mask_i))

    # check for corner case of no geoms
    if len(masks) == 0:
        masks.append(np.zeros((img.meta["height"], img.meta["width"])))
        p.append(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))

    # crop to the same shape as img, if slightly off
    masks = np.stack(masks)
    if masks.shape[1] != img.meta["height"] or masks.shape[2] != img.meta["width"]:
        masks = masks[:, :img.meta["height"], :img.meta["width"]]

    return masks, p, y_extent


def preprocessor(img, y):
    """
    Helper for processing x, y for levelsets

    Inputs
    x: rasterio image object, with bounds
    y: geopandas data.frame used to create training data masks
    """
    x = img.read()
    y, extreme_polys, y_extent = mask(y, img)
    y_init = reverse_buffer(y_extent, img)

    dist, signed_dist = sdt(y)
    extreme_hm = gaussian_convolve(x.shape[1:], np.vstack(extreme_polys))
    gradient = [inverse_gaussian_gradient(x).mean(axis=0)]
    maxes = [z.max(0) for z in [y, extreme_hm]]
    mins = [z.min(0) for z in [dist, signed_dist]]
    meta = [maxes[1]] + mins + gradient
    meta = [(s - s.mean()) / s.std() for s in meta]
    meta.append(y_init.squeeze())

    meta = np.stack(meta)
    y = maxes[0][np.newaxis, ...]
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

    if len(fn) != len(meta_fn):
        raise ArgumentError("The number of files in --meta_dir and --x_dir must be equal, so that image and metadata can be matched.")

    out_fn_y = [infer_dir / (f.stem + "_pred.tif") for f in fn]
    out_fn_prob = [infer_dir / (f.stem + "_prob.tif") for f in fn]

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


def list_ids(source_dir):
    fn = list(pathlib.Path(source_dir).glob("**/*tif"))
    return pd.DataFrame({
        "path": fn,
        "sample_id": [str(f.stem) for f in fn]
    })


def eval_paths(ids, inference_dir, mode="pred"):
    fn = [pathlib.Path(inference_dir) / f"{s}_{mode}.tif" for s in ids]
    return pd.DataFrame({
        "path": fn,
        "sample_id": ids
    })


def process_scene(scene, y, in_dir, stat_path):
    img = rasterio.open(scene)
    mean, std, label, meta = preprocessor(img, y)
    save_raster(label, img.meta, img.transform, in_dir / f"labels/{scene.stem}-labels.tif")
    save_raster(meta, img.meta, img.transform, in_dir / f"meta/{scene.stem}-meta.tif")

    f = open(stat_path, "a")
    writer = csv.writer(f)
    writer.writerow([str(scene.stem)] + list(np.hstack([mean, std])))


def preprocess_dir(in_dir, y, n_jobs=20):
    if (in_dir / "images").exists():
        shutil.rmtree(in_dir / "images")
        shutil.rmtree(in_dir / "labels")
    (in_dir / "images").mkdir(parents=True)
    (in_dir / "meta").mkdir(parents=True)

    scene_list = list(in_dir.glob("*.tif"))
    tmp = rasterio.open(scene_list[0]).read()
    fields = ["scene"] + sum([[f"{s}_{i}" for i in range(tmp.shape[0])] for s in ["mean", "sdev"]], [])
    stat_path = in_dir / "statistics.csv"
    f = open(stat_path, "a")
    writer = csv.writer(f)
    writer.writerow(fields)
    f.flush()

    print(f"preprocessing {in_dir}")
    Parallel(n_jobs=n_jobs)(
        delayed(process_scene)(fn, y, in_dir, stat_path) for fn in tqdm(scene_list)
    )

    f.close()
    [shutil.move(str(s), in_dir / "images") for s in scene_list]

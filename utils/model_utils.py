import sys
sys.path.append("..")
from data.dataloader import image_transforms
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter
from shapely.geometry import box
from torch import nn
from torch.nn import functional as F
from utils.data import save_raster, mask
import geopandas as gpd
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import rasterio
import rasterio.features as rf
import shutil
import tempfile
import torch
import utils.metrics as mt
import sys
sys.path.append("..")
from data.dataloader import image_transforms
from scipy.ndimage import gaussian_filter
import fiona
from shapely.geometry import box
from shapely.geometry import Polygon

class LocalContextNorm(nn.Module):
    def __init__(self, num_features, channels_per_group=2, window_size=(227, 227), eps=1e-5):
        super(LocalContextNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.channels_per_group = channels_per_group
        self.eps = eps
        self.window_size = window_size

    def forward(self, x):
        N, C, H, W = x.size()
        G = C // self.channels_per_group
        assert C % self.channels_per_group == 0
        if self.window_size[0] < H and self.window_size[1] < W:
            # Build integral image
            device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
            x_squared = x * x
            integral_img = x.cumsum(dim=2).cumsum(dim=3)
            integral_img_sq = x_squared.cumsum(dim=2).cumsum(dim=3)
            # Dilation
            d = (1, self.window_size[0], self.window_size[1])
            integral_img = torch.unsqueeze(integral_img, dim=1)
            integral_img_sq = torch.unsqueeze(integral_img_sq, dim=1)
            kernel = torch.tensor([[[[[1., -1.], [-1., 1.]]]]]).to(device)
            c_kernel = torch.ones((1, 1, self.channels_per_group, 1, 1)).to(device)
            with torch.no_grad():
                # Dilated conv
                sums = F.conv3d(integral_img, kernel, stride=[1, 1, 1], dilation=d)
                sums = F.conv3d(sums, c_kernel, stride=[self.channels_per_group, 1, 1])
                squares = F.conv3d(integral_img_sq, kernel, stride=[1, 1, 1], dilation=d)
                squares = F.conv3d(squares, c_kernel, stride=[self.channels_per_group, 1, 1])
            n = self.window_size[0] * self.window_size[1] * self.channels_per_group
            means = torch.squeeze(sums / n, dim=1)
            var = torch.squeeze((1.0 / n * (squares - sums * sums / n)), dim=1)
            _, _, h, w = means.size()
            pad2d = (int(math.floor((W - w) / 2)), int(math.ceil((W - w) / 2)), int(math.floor((H - h) / 2)),
                     int(math.ceil((H - h) / 2)))
            padded_means = F.pad(means, pad2d, 'replicate')
            padded_vars = F.pad(var, pad2d, 'replicate') + self.eps
            for i in range(G):
                x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] = \
                    (x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] -
                     torch.unsqueeze(padded_means[:, i, :, :], dim=1).to(device)) /\
                    ((torch.unsqueeze(padded_vars[:, i, :, :], dim=1)).to(device)).sqrt()
            del integral_img
            del integral_img_sq
        else:
            x = x.view(N, G, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            x = (x - mean) / (var + self.eps).sqrt()
            x = x.view(N, C, H, W)

        return x * self.weight + self.bias


class CheckpointSaver(object):
    def __init__(self, save_dir, backup_dir):
        self.save_dir = save_dir
        self.backup_dir = backup_dir

    def save(self, state, is_best, checkpoint_name='checkpoint'):
        checkpoint_path = os.path.join(self.save_dir,
                                       '{}.pth.tar'.format(checkpoint_name))
        try:
            shutil.copyfile(
                checkpoint_path,
                '{}_bak'.format(checkpoint_path)
            )
        except IOError:
            pass
        torch.save(state, checkpoint_path)
        if is_best:
            try:
                shutil.copyfile(
                    os.path.join(self.backup_dir,
                                 '{}_best.pth.tar'.format(checkpoint_name)),
                    os.path.join(self.backup_dir,
                                 '{}_best.pth.tar_bak'.format(checkpoint_name))
                )
            except IOError:
                pass
            shutil.copyfile(
                checkpoint_path,
                os.path.join(self.backup_dir,
                             '{}_best.pth.tar'.format(checkpoint_name))
            )


def save_loss(train_loss, val_loss, save_dir, name='loss_plots'):
    """

    :param train_loss: train losses in different epochs
    :param val_loss: validation losses in different epochs
    :return:
    """
    plt.yscale('log')
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper right')
    plt.savefig(save_dir + name + '.png')


# Function to load model options
def load_options(file_name):
    if file_name.endswith(".pkl"):
        opt = pickle.load(open(file_name, 'rb'))
    else:
        opt = pickle.load(open(file_name + '.pkl', 'rb'))
    return opt

def prepare_tile(fn, meta_fn, stride, chip_size):
    x = rasterio.open(fn).read()
    meta = rasterio.open(meta_fn).read()
    x = np.moveaxis(x, 0, 2)
    meta = np.moveaxis(meta, 0, 2)
    dim = x.shape

    # pad the data
    pad = pad_size(x.shape, chip_size, stride)
    x = np.pad(x, pad)
    meta = np.pad(meta, pad)
    return x, meta, dim, Path(fn).stem


# generic inference function
def inference_gen(pred_fun, processor, stride=150, chip_size=256):
    def inferencer(fn, meta_fn, stats_fn):
        x, meta, dim, id = prepare_tile(fn, meta_fn, stride, chip_size)
        with torch.no_grad():
            sweep_ix = sweep_indices(dim, stride, chip_size)
            y_hat, probs = inference_sweep(
                x, meta, stats_fn, id,
                pred_fun, processor, sweep_ix
            )

        return y_hat[None, :dim[0], :dim[1]], probs[None, :dim[0], :dim[1]]
    return inferencer

def cpu(z):
    return z.cpu().numpy()

def inference_sweep(x, meta, stats_fn, id, pred_fun, processor, sweep_ix):
    y_hat, probs, counts = [np.zeros(x.shape[:2]) for _ in range(3)]
    for i, (h, w) in enumerate(sweep_ix):
        x_, meta_ = processor(x[h, w], meta[h, w], stats_fn, id)
        y_hat_, probs_, _ = pred_fun(x_, meta_)
        y_hat[h, w] += cpu(y_hat_[0])
        probs[h, w] += cpu(probs_[0, 0])
        counts[h, w] += 1

    probs /= counts
    y_hat /= counts
    return 1 * (y_hat > 0.5), probs


def pad_size(dim, chip_size, stride=None):
    if stride is None:
        stride = chip_size

    pad = [stride * (dim[i] // stride) + chip_size - dim[i] for i in range(2)]
    return [(0, s) for s in pad + [0]]


def sweep_indices(dim, stride, chip_size):
    ix = [np.arange(0, dim[i], stride) for i in range(2)]
    result = []

    for h in ix[0]:
        for w in ix[1]:
            result.append((
                slice(int(h), int(h) + chip_size),
                slice(int(w), int(w) + chip_size)
            ))

    return result


# example input pre and postprocessing functions for inference
def processor_chip(device):
    def f(x, meta, stats_fn, id):
        x_ = image_transforms(x, stats_fn, id).to(device).unsqueeze(0)
        meta = np.moveaxis(meta, 2, 0)
        meta_ = torch.from_numpy(meta).to(device).unsqueeze(0)
        return x_, meta_
    return f


def blur_raster(x, sigma=2, threshold=0.5):
    blurred = gaussian_filter(x.read(), sigma=sigma)
    f = tempfile.NamedTemporaryFile()
    save_raster(blurred > threshold, x.meta, x.transform, Path(f.name))
    return rasterio.open(Path(f.name))


def polygonize_preds(y_hat, crop_region, tol=25e-5):
    # if no polygon, just return the center of the prediction region
    centroid = gpd.GeoDataFrame(geometry=[box(*y_hat.bounds).centroid])

    # get features from probability and overlay onto crop region
    ft = list(rf.dataset_features(blur_raster(y_hat), as_mask=True))
    ft = gpd.GeoDataFrame.from_features(ft)
    if len(ft) == 0:
        return centroid

    crop_region = gpd.GeoDataFrame(geometry=[crop_region])
    result = gpd.overlay(ft, crop_region).simplify(tolerance=tol)

    if len(result) == 0:
        return centroid
    return gpd.GeoDataFrame(geometry=result)


def polygon_metrics(y_hat, y, context, metrics={"IoU": mt.IoU}):
    y_, _ = mask(y, context)
    y_hat_, _ = mask(y_hat, context)

    y_ = y_.sum(axis=0, keepdims=True)
    y_hat_ = y_hat_.sum(axis=0, keepdims=True)
    return {k: m(y_hat_, y_).item() for k, m in metrics.items()}

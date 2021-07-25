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


# generic inference function
def inference_gen(pred_fun, processor, postprocessor, **kwargs):
    def inferencer(fn, meta_fn, stats_fn):
        x, meta, pre = processor(fn, meta_fn, stats_fn, **kwargs)
        with torch.no_grad():
            y_hat, probs, _ = pred_fun(x, meta)

        return postprocessor(y_hat, probs, pre, **kwargs)
    return inferencer


# example input pre and postprocessing functions for inference
def processor_test(fn, meta_fn, stats_fn, device, out=(1024, 1024), **kwargs):
    x = rasterio.open(fn).read()
    meta = rasterio.open(meta_fn).read()
    x = np.transpose(x, (1, 2, 0))
    x_ = np.pad(x, ((0, out[0] - x.shape[0]), (0, out[1] - x.shape[1]), (0, 0)))

    id = Path(fn).stem
    x_ = image_transforms(x_, stats_fn, id).to(device).unsqueeze(0)
    return x_, meta, {"dim": x.shape}


def postprocessor_test(y_hat, probs, pre, **kwargs):
    y_hat = y_hat[:, :pre["dim"][0], :pre["dim"][1]]
    probs = probs[:, :, :pre["dim"][0], :pre["dim"][1]]
    cpu = lambda x: x.cpu().numpy().squeeze()
    return cpu(y_hat)[np.newaxis], cpu(probs)


def blur_raster(x, sigma=2, threshold=0.5):
    blurred = gaussian_filter(x.read(), sigma=sigma)
    f = tempfile.NamedTemporaryFile()
    save_raster(blurred > threshold, x.meta, x.transform, Path(f.name))
    return rasterio.open(Path(f.name))


def polygonize_preds(y_hat, crop_region, tol=25e-5):
    # get features from probability and overlay onto crop region
    ft = list(rf.dataset_features(blur_raster(y_hat), as_mask=True))
    ft = gpd.GeoDataFrame.from_features(ft)
    crop_region = gpd.GeoDataFrame(geometry=[crop_region])
    result = gpd.overlay(crop_region, ft).simplify(tolerance=tol)

    # if no polygon, just return the center of the prediction region
    if len(result) == 0:
        return gpd.GeoDataFrame(geometry=[box(*y_hat.bounds).centroid])
    return gpd.GeoDataFrame(geometry=result)


def polygon_metrics(y_hat, y, context, metrics={"IoU": mt.IoU}):
    y_, _ = mask(y, context)
    y_hat_, _ = mask(y_hat, context)

    y_ = y_.sum(axis=0, keepdims=True)
    y_hat_ = y_hat_.sum(axis=0, keepdims=True)
    return {k: v(y_hat_, y_) for k, v in metrics.items()}

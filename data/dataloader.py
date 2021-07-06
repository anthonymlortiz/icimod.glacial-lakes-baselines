from data.streaming_dataset import StreamingGeospatialDataset
import torch
from utils import utils
import numpy as np
import glob
import pandas as pd
import os


def get_stats_fn(base_dir, split, dataset):
    return get_fns_from_suffix(base_dir, split, dataset, "statistics.csv")


def get_imagery_fns(base_dir, split, dataset):
    return get_fns_from_suffix(base_dir, split, dataset, "/images/*.tif")


def get_labels_fns(base_dir, split, dataset):
    return get_fns_from_suffix(base_dir, split, dataset, "/labels/*.tif")


def get_fns_from_suffix(base_dir, split, dataset, suffix):
    if suffix == "statistics.csv":
        if dataset.lower() == 'bing':
            return os.path.join(base_dir+"/bing/splits/"+split+"/", suffix)
        if dataset.lower() == 'landsat':
            return os.path.join(base_dir+"/le7-2015/splits/"+split+"/", suffix)
        if dataset.lower() == 'maxar':
            return os.path.join(base_dir+"/maxar/splits/"+split+"/", suffix)
        else:
            raise ValueError("Dataset %s is not recognized" % (dataset))
    else:   
        if dataset.lower() == 'bing':
            return glob.glob(base_dir+"/bing/splits/"+split + suffix)
        if dataset.lower() == 'landsat':
            return glob.glob(base_dir+"/le7-2015/splits/"+split + suffix)
        if dataset.lower() == 'maxar':
            return glob.glob(base_dir+"/maxar/splits/"+split+ suffix)
        else:
            raise ValueError("Dataset %s is not recognized" % (dataset))


def get_image_transforms():
    """
    docstring
    """
    return image_transforms


def image_transforms(img, stats_fn, id):
    mean, std = get_imagery_statistics(stats_fn, id)
    img = (img - mean) / std
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img


def get_imagery_statistics(stats_fn,id):
    df = pd.read_csv(stats_fn, index_col=0)
    means = np.array([df["mean_0"][id], df["mean_1"][id], df["mean_2"][id], df["mean_3"][id], df["mean_4"][id],
                     df["mean_5"][id], df["mean_6"][id], df["mean_7"][id], df["mean_8"][id], df["mean_9"][id], 
                     df["mean_10"][id]])
    stds = np.array([df["sdev_0"][id], df["sdev_1"][id], df["sdev_2"][id], df["sdev_3"][id], df["sdev_4"][id], 
                    df["sdev_5"][id], df["sdev_6"][id], df["sdev_7"][id], df["sdev_8"][id], df["sdev_9"][id], 
                    df["sdev_10"][id]])
    return means, stds



def load_dataset(opts, kwargs=None):
    img_transforms = get_image_transforms()

    train_img_fns = get_imagery_fns(opts.data_dir, "train", opts.dataset)
    train_label_fns = get_labels_fns(opts.data_dir, "train", opts.dataset)
    stats_fn = get_stats_fn(opts.data_dir, "train", opts.dataset)
    trn = StreamingGeospatialDataset(train_img_fns, stats_fn, train_label_fns, train_img_fns, chip_size=opts.chip_size, num_chips_per_tile=20, image_transform=img_transforms, verbose=False)
              
    val_img_fns = get_imagery_fns(opts.data_dir, "val", opts.dataset)
    val_label_fns = get_labels_fns(opts.data_dir , "val", opts.dataset)
    val = StreamingGeospatialDataset(val_img_fns, stats_fn, val_label_fns, train_img_fns, chip_size=opts.chip_size, num_chips_per_tile=10, image_transform=img_transforms, verbose=False)

    trn_loader = torch.utils.data.DataLoader(trn, batch_size=opts.batch_size, num_workers=opts.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=opts.batch_size, num_workers=opts.num_workers, pin_memory=True)

    dataloaders = {'train': trn_loader, 'val': val_loader}
    return dataloaders
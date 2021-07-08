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


def get_meta_fns(base_dir, split, dataset):
    return get_fns_from_suffix(base_dir, split, dataset, "/meta/*.tif")


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


def image_transforms(img, stats_fn, id): #, stats_fn, id):
    mean, std = get_imagery_statistics(stats_fn, id)
    img = (img - mean) / std
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = np.nan_to_num(img)
    img = torch.from_numpy(img)
    return img


def get_imagery_statistics(stats_fn, id):
    df = pd.read_csv(stats_fn, index_col=0)
    means = df.loc[id].filter(like="mean").values
    stds = df.loc[id].filter(like="sdev").values
    return means, stds


def load_dataset(opts, kwargs=None):

    img_transforms = get_image_transforms()
    train_img_fns = get_imagery_fns(opts.data_dir, "train", opts.dataset)
    train_label_fns = get_labels_fns(opts.data_dir, "train", opts.dataset)
    train_meta_fns = get_meta_fns(opts.data_dir , "train", opts.dataset)
    train_img_fns.sort()
    train_label_fns.sort()
    train_meta_fns.sort()
    stats_fn = get_stats_fn(opts.data_dir, "train", opts.dataset)
    trn = StreamingGeospatialDataset(train_img_fns, stats_fn, train_label_fns,train_meta_fns,  groups=train_img_fns, label_channels=opts.label_channels, chip_size=opts.chip_size, num_chips_per_tile=20, image_transform=img_transforms, verbose=False)

    val_img_fns = get_imagery_fns(opts.data_dir, "val", opts.dataset)
    val_label_fns = get_labels_fns(opts.data_dir , "val", opts.dataset)
    val_meta_fns = get_meta_fns(opts.data_dir , "val", opts.dataset)
    val_img_fns.sort()
    val_label_fns.sort()
    val = StreamingGeospatialDataset(val_img_fns, stats_fn, val_label_fns, val_meta_fns, groups=val_img_fns, chip_size=opts.chip_size, num_chips_per_tile=10, image_transform=img_transforms, verbose=False)

    trn_loader = torch.utils.data.DataLoader(trn, batch_size=opts.batch_size, num_workers=opts.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=opts.batch_size, num_workers=opts.num_workers, pin_memory=True)

    dataloaders = {'train': trn_loader, 'val': val_loader}
    return dataloaders

import sys
import shutil
import os
import torch
import utils.metrics as mt
from trainers import train_funs
from trainers.train_framework import Algorithm
from models.losses import MulticlassCrossEntropy, WeightedBCELoss
from options.train_options import TrainOptions
from data.dataloader import load_dataset
from models.unet import UnetModel
from tensorboardX import SummaryWriter
from pathlib import Path
from warnings import warn, filterwarnings
filterwarnings("ignore", category=UserWarning)


# parse options
opts = TrainOptions().parse()
print(' '.join(sys.argv))

# Define model according to opts
if opts.model == "unet":
    model = UnetModel(opts)
else:
    assert NotImplementedError, f"Option {opts.model} not supported. Available options: unet,fcn, hrnet"
model = model.to(torch.device(opts.device))

# Define loss function or criterion based on opts
if opts.loss == "ce":
    loss = MulticlassCrossEntropy()
elif opts.loss == "wbce":
    loss = WeightedBCELoss()
else:
    assert NotImplementedError, f"Option {opts.loss} not supported. Available options: ce, wbce"

# Setup optimizer according to opts
if opts.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
if opts.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9)


if opts.overwrite:
    warn("You have chosen to overwrite previous training directory for this experiment")
    shutil.rmtree(opts.save_dir + "/" + opts.experiment_name + "/training")
    os.makedirs(opts.save_dir + "/" + opts.experiment_name + "/training")

metrics = {"IoU": mt.IoU, "precision": mt.precision, "recall": mt.recall}
frame = Algorithm(model, loss, optimizer, metrics, opts)
datasets = load_dataset(opts)
writer = SummaryWriter(Path(opts.save_dir) / opts.experiment_name)
train_funs.train(frame, datasets, writer, opts)

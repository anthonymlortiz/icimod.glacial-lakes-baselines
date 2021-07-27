import sys
import shutil
import os
import torch
import utils.metrics as mt
from trainers import train_funs
from trainers.algorithm import Algorithm
from trainers.delse_algo import DelseAlgo
from models import losses
from options.train_options import TrainOptions
from data.dataloader import load_dataset
from models.unet import UnetModel
from models.delse import DelseModel
from tensorboardX import SummaryWriter
from pathlib import Path
from warnings import warn, filterwarnings
filterwarnings("ignore", category=UserWarning)


# parse options
opts = TrainOptions().parse()
opts.save_dir = Path(opts.save_dir)
opts.backup_dir = Path(opts.backup_dir)
opts.log_dir = Path(opts.log_dir)
opts.save_dir.mkdir(parents=True, exist_ok=True)
opts.backup_dir.mkdir(parents=True, exist_ok=True)
opts.log_dir.mkdir(parents=True, exist_ok=True)

# Define model according to opts
if opts.model == "unet":
    model = UnetModel(opts)
    params = model.parameters()
elif opts.model == "delse":
    model = DelseModel(opts)
    params = [{'params': model.full_model.get_1x_lr_params(), 'lr': opts.lr},
              {'params': model.full_model.get_10x_lr_params(), 'lr': opts.lr * 10}]
else:
    assert NotImplementedError, f"Option {opts.model} not supported. Available options: unet,delse"
model = model.to(torch.device(opts.device))

# Define loss function or criterion based on opts
if opts.loss == "ce":
    loss = losses.MulticlassCrossEntropy()
elif opts.loss == "wbce":
    loss = losses.WeightedBCELoss()
elif opts.loss == "delse":
    loss = losses.DelseLoss(opts.delse_epsilon)
else:
    assert NotImplementedError, f"Option {opts.loss} not supported. Available options: ce, wbce"

# Setup optimizer according to opts
if opts.optimizer == "adam":
    optimizer = torch.optim.Adam(params, lr=opts.lr, betas=(opts.beta1, opts.beta2))
if opts.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9)

metrics = {"IoU": mt.IoU, "precision": mt.precision, "recall": mt.recall}
if opts.model == "unet":
    frame = Algorithm(model, loss, optimizer, metrics, opts)
elif opts.model == "delse":
    frame = DelseAlgo(model, loss, optimizer, metrics, opts)

datasets = load_dataset(opts)
writer = SummaryWriter(opts.log_dir / opts.experiment_name)
train_funs.train(frame, datasets, writer, opts)

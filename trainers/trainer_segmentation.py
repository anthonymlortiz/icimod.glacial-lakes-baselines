import torch
import torch.nn as nn
import os, sys
from time import time
from utils import metrics
from torch.optim import lr_scheduler
import numpy as np
from utils.model_utils import save_loss, CheckpointSaver
from tqdm import tqdm
import pandas as pd


def train_epoch(algorithm, dataset):
    iterator = tqdm(dataset)
    algorithm.model.train()
    for batch in iterator:
        algorithm.update(batch)


def evaluate(model_fun, batch, metrics, device):
    x, y = [s.to(device) for s in batch]
    y_pred, outputs = model_fun(x)
    metrics_ = {k: m(y_pred, y) for k, m in metrics.items()}
    return metrics_, y, outputs


def train_(algorithm, datasets, writer, opts, epoch_start=0, best_val=None):
    for epoch in range(epoch_start, opts.n_epochs):
        #writer.add_text(f"Starting Epoch:\n", epoch, time())
        train_epoch(algorithm, datasets["train"])

        #writer.add_text("Starting validation", epoch, time())
        metrics = {
            "val": validate(algorithm, datasets["val"]),
            "train": validate(algorithm, datasets["train"])
        }

        log_epoch(writer, epoch, metrics)
        best_val = save_if_needed(algorithm, metrics, best_val, epoch, opts)


def validate(algorithm, dataset):
    algorithm.model.eval()
    metrics, objective = [], []
    for batch in dataset:
        print("evaluating")
        metrics_, y, outputs = evaluate(
                algorithm.model.infer,
                batch,
                metrics,
                algorithm.device
            )
        objective.append(algorithm.objective(y, outputs))
        metrics.append(metrics_)

    import pdb
    pdb.set_trace()
    metrics = pd.DataFrame(metrics)
    return {
        "avg": metrics.mean(axis=0),
        "sample": metrics,
        "objective": np.mean(objective)
    }


def log_epoch(writer, epoch, metrics):
    #writer.add_text("Completed validation", epoch, time())
    for split in ["val", "train"]:
        writer.add_scalar(f"Obj/{split}", metrics[split]["objective"], epoch)
        for m in metrics[split]["avg"].index:
            writer.add_scalar(f"{m}/{split}", metrics[split]["avg"][m], epoch)


def save_if_needed(algorithm, metrics, best_val, epoch, opts):
    if best_val is None or metrics[opts.val_metric] < best_val:
        print(metrics["val"]["avg"])
        best_val = metrics["val"]["avg"][opts.val_metric]
        algorithm.save(epoch, opts)
    elif epoch % opts.save_epoch == 0:
        algorithm.save(epoch, opts, str(epoch))
    return best_val

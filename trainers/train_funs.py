import numpy as np
from tqdm import tqdm
import pandas as pd


def train_epoch(algorithm, dataset):
    algorithm.model.train()
    for batch in tqdm(dataset):
        algorithm.update(batch)


def evaluate(model_fun, batch, metrics, device):
    x, y, meta = [s.to(device) for s in batch]
    y_pred, outputs = model_fun(x, meta)
    metrics_ = {k: m(y_pred, y).cpu().numpy() for k, m in metrics.items()}
    return metrics_, y, outputs, meta


def train(algorithm, datasets, writer, opts, epoch_start=0, best_val=None):
    for epoch in range(epoch_start, opts.n_epochs):
        train_epoch(algorithm, datasets["train"])
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
        metrics_, y, outputs, meta = evaluate(
                algorithm.model.infer,
                batch,
                algorithm.metrics,
                algorithm.device
            )
        objective.append(algorithm.objective(y, outputs, meta).item())
        metrics.append(pd.DataFrame(metrics_))

    metrics = pd.concat(metrics)
    return {
        "avg": metrics.mean(axis=0),
        "sample": metrics,
        "objective": np.mean(objective)
    }


def log_epoch(writer, epoch, metrics):
    for split in ["val", "train"]:
        writer.add_scalar(f"Obj/{split}", metrics[split]["objective"], epoch)
        for m in metrics[split]["avg"].index:
            writer.add_scalar(f"{m}/{split}", metrics[split]["avg"][m], epoch)


def save_if_needed(algorithm, metrics, best_val, epoch, opts):
    cur_val = metrics["val"]["avg"][opts.val_metric]
    if best_val is None or cur_val < best_val:
        best_val = cur_val
        algorithm.save()
    elif epoch % opts.save_epoch == 0:
        algorithm.save(str(epoch))
    return best_val

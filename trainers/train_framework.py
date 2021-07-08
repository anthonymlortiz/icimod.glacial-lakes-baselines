import torch
from torch.nn.utils import clip_grad_norm_
from pathlib import Path


class Algorithm:

    def __init__(self, model, loss, optimizer, metrics, opts):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.device = torch.device(opts.device)
        self.max_grad_norm = opts.max_grad_norm
        self.opts = opts

    def process_batch(self, batch):
        x, y = [s.to(self.device) for s in batch]
        outputs = self.model(x)
        return y, outputs

    def objective(self, y, outputs):
        return self.loss(outputs, y)

    def update(self, batch):
        y, outputs = self.process_batch(batch)
        objective = self._update(y, outputs)
        return y, outputs, objective

    def _update(self, y, outputs):
        objective = self.objective(y, outputs)
        self.optimizer.zero_grad()
        objective.backward()

        if self.max_grad_norm:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        return objective.item()

    def save(self, suffix="best"):
        fname = f"{self.opts.experiment_name}_{suffix}.pth"
        torch.save(self.model.state_dict(), Path(self.opts.save_dir) / fname)

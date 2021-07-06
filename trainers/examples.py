import torch
from train_framework import Algorithm
from utils import lse
import numpy as np


class DelseAlgo(Algorithm):
    def __init__(self, model, loss, metrics, opts):
        super().__init__(model=model, loss=loss, metrics=metrics, opts=opts)
        self.epoch = opts.epoch
        self.pretrain_epoch = opts.pretrain_epoch
        self.T = opts.T
        self.dt_max = opts.dt_max
        self.epsilon = opts.epsilon
        self.shift = True

    def process_batch(self, batch):
        y, outputs = super().process_batch(batch)
        sdt = y[:, 1]  # ?
        phi_0, energy, g = [lse.interpolater(z, sdt.shape[-1]) for z in outputs]
        g = torch.sigmoid(g)
        vfs = lse.gradient(y, split=False)

        if self.shift:
            sdt += 10 * np.random.rand() - 5

        phi_T = lse.levelset_evolution(sdt, energy, g, self.T, self.dt_max)
        outputs = {"y_pred": phi_T, "vfs": vfs, "energy": energy, "g": g,
                   "sdt": sdt, "phi_0": phi_0}
        return y, outputs

    def objective(self, y, outputs):
        li = self.loss(outputs["phi_0"], outputs["sdt"], outputs["energy"],
                       outputs["vfs"], outputs["y_pred"], y)
        if self.epoch > self.pretrain_epochs:
            li[:2] = 0

        return sum(li)

    def update(self, batch):
        self.epoch += 1
        super().update(self, batch)

    def evaluate(self, batch):
        y, outputs = self.process_batch(batch)
        objective = self.objective(y, outputs).item()
        return y, outputs["y_pred"], objective

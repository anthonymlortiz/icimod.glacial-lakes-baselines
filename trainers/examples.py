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
        return y, [phi_0, sdt, energy, vfs, phi_T]

    def objective(self, y, outputs):
        li = self.loss(y, **outputs)
        if self.epoch > self.pretrain_epochs:
            li[:2] = 0

        return sum(li)

    def update(self, batch):
        self.epoch += 1
        super().update(self, batch)

    def process_output(self, outputs):
        return outputs[-1]

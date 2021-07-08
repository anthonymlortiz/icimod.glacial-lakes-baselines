from train_framework import Algorithm
from utils import lse
import numpy as np


class DelseAlgo(Algorithm):
    def __init__(self, model, loss, metrics, opts):
        super().__init__(model=model, loss=loss, metrics=metrics, opts=opts)
        self.epoch = opts.epoch
        self.pretrain_epoch = opts.pretrain_epoch
        self.epsilon = opts.epsilon
        self.shift = opts.shift

    def objective(self, batch):
        y, (phi_0, energy, g) = self.process_batch(batch)
        sdt = y[:, 1, :]
        vfs = lse.gradient(y, split=False)
        shift = 10 * np.random.rand() - 5

        # compute evolution
        if self.epoch < self.pretrain_epoch:
            phi_T = lse.levelset_evolution(sdt + shift, energy, g,
                                           self.model.T, self.model.dt_max)
        else:
            phi_T = lse.levelset_evolution(phi_0 + shift, energy, g,
                                           self.model.T, self.model.dt_max)

        # return losses
        losses = self.loss(y, phi_0, sdt, energy, vfs, phi_T)
        if self.epoch > self.pretrain_epoch:
            losses = losses[-1]
        return sum(losses)

    def update(self, batch):
        self.epoch += 1
        super().update(self, batch)

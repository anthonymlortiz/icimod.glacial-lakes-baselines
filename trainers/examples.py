from trainers.train_framework import Algorithm
from utils import lse
import numpy as np


class DelseAlgo(Algorithm):
    def __init__(self, model, loss, optimizer, metrics, opts):
        super().__init__(model, loss, optimizer, metrics, opts)
        self.epoch = 0
        self.epsilon = opts.delse_epsilon
        self.model = model
        self.pretrain_epoch = opts.delse_pretrain

    def objective(self, y, outputs, meta):
        y = y.unsqueeze(1)
        (phi_0, energy, g) = outputs
        sdt = meta[:, 2:3]  # signed distance transform
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
            losses = [losses[-1]]
        return sum(losses)

    def update(self, batch):
        self.epoch += 1
        super().update(batch)

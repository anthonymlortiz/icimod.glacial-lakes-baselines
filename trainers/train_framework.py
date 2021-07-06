import torch
from torch.nn.utils import clip_grad_norm_
from pathlib import Path


class TrainFramework:
    def __init__(self, model, loss, opts):
        self.model = model
        self.loss = loss
        self.lr = opts.lr
        self.beta1 = opts.beta1
        self.beta2 = opts.beta2
        if opts.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-08, weight_decay=0, amsgrad=False)
        elif opts.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr,
                                momentum=0.9,
                                weight_decay=1e-5)
        else:
            print("Option {} not supported. Available options: adam, sgd".format(opts.optimizer))
            raise NotImplementedError

        if torch.cuda.is_available():
            self.model = model.cuda()
            if opts.loss != "lcfcn":
                self.loss = loss.cuda()

    def optimize(self, X, y):
        y_pred = self.model.forward(X)
        loss = self.loss(y, y_pred)
        return loss, y_pred

    def backwardpass(self, loss):
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)


class Algorithm:

    def __init__(self, model, loss, metrics, opts):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.device = opts.device
        self.max_grad_norm = opts.max_grad_norm

    def process_batch(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        outputs = self.model(x)
        return y, outputs

    def process_output(self, outputs):
        """Convert output to y_pred"""
        return outputs

    def evaluate(self, batch):
        y, outputs = self.process_batch(batch)
        objective = self.loss(y, outputs).item()
        y_pred = self.process_outputs(outputs)

        result = {}
        for k, metric in self.metrics:
            result[k] = metric(y_pred, y)
        return result, objective

    def update(self, batch):
        y, outputs = self.process_batch(batch)
        objective = self._update(y, outputs)
        return y, outputs, objective

    def _update(self, y, outputs):
        objective = self.objective(y, outputs)
        self.model.zero_grad()
        objective.backward()

        if self.max_grad_norm:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        return objective.item()

    def update_log(self, y, outputs, objective):
        pass

    def save_model(self, suffix="best"):
        fname = f"{self.opts.id}-{suffix}.pth"
        torch.save(self.model.state_dict(), Path(self.opts.out_dir) / fname)

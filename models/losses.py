import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append("..")
from utils import lse


class MulticlassCrossEntropy(nn.Module):
    """Multiclass crossentropy loss Pytorch (no one hot)

    Args:
        y_pred: Model predictions
        y_true: Ground truth labels
    """

    def __init__(self, batch=True):
        super(MulticlassCrossEntropy, self).__init__()
        self.batch = batch
        self.crossentropy = nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        loss = self.crossentropy(y_pred, y_true)
        return loss


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        logpt = -F.cross_entropy(input, target, Variable(self.weight))
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WeightedBCELoss(nn.Module):
    """Weighted binary crossentropy, by default weights are [0.5, 0.5],
    equivalent to standard BCE

    Args:
        nn ([type]): [description]
    """

    def __init__(self, weights=[0.2, 0.8], batch=True):
        super(WeightedBCELoss, self).__init__()
        self.batch = batch
        self.class_weights = torch.FloatTensor(weights).cuda()
        self.bce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

    def __call__(self, y_pred, y_true):
        loss = self.bce_loss(y_pred, y_true.long())
        return loss


class DiceBCELoss(nn.Module):
    """DiceBCELoss is a joint BCE and Dice loss optimizing for both objectives

    Args:
        nn ([type]): [description]
    """

    def __init__(self, batch=True):
        super(DiceBCELoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.CrossEntropyLoss()

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.001  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_pred * y_true)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        y_true = y_true.long()
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred[:, 1], y_true)
        return a + b


class DiceLoss(nn.Module):
    """Dice loss Pytorch implementation

    Args:
        nn ([type]): [description]
    """

    def __init__(self, batch=True):
        super(DiceLoss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.001  # may
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_pred * y_true)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred, y_true)


class DelseLoss(nn.Module):
    """Loss used in DELSE
    """

    def __init__(self, epsilon=1, alpha=1, eta=100):
        super(DelseLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.eta = eta

    def __call__(self, y, phi_0, sdt, energy, vfs, phi_T):

        return [lse.mean_square_loss(phi_0, sdt, self.alpha),
                lse.vector_field_loss(energy, vfs),
                lse.LSE_loss(phi_T, y, sdt, self.epsilon, self.eta)]

from torch.nn.utils import clip_grad_norm_
import lse
import torch
import numpy as np
import torch.nn.functional as F


def mean_square_loss(y_hat, y, alpha=1):
    assert (y_hat.size() == y.size())
    mse = lambda x: torch.mean(torch.pow(x, 2))
    sdt_loss = mse(y - y_hat)
    return alpha * sdt_loss


def balanced_bce(outputs, labels):
    assert(outputs.size() == labels.size())
    labels_count = (torch.sum(labels), torch.sum(1.0 - labels))
    N = torch.numel(labels)
    loss_val = -torch.mul(labels, torch.log(outputs)) -\
        torch.mul((1.0 - labels), torch.log(1.0 - outputs))

    loss_pos = torch.sum(torch.mul(labels, loss_val))
    loss_neg = torch.sum(torch.mul(1.0 - labels, loss_val))
    final_loss = labels_count[1] / N * loss_pos + labels_count[0] / N * loss_neg
    return final_loss / N


def vector_field_loss(vf_pred, vf_gt):
    # (n_batch, n_channels, H, W)
    vf_pred = F.normalize(vf_pred, p=2, dim=1)
    vf_gt = F.normalize(vf_gt.float(), p=2, dim=1)
    cos_dist = torch.sum(torch.mul(vf_pred, vf_gt), dim=1)
    angle_error = torch.acos(cos_dist * (1-1e-4))
    return torch.mean(torch.pow(angle_error, 2))


def LSE_loss(phi_T, gts, sdts, epsilon=-1, eta=100):
    return eta * balanced_bce(lse.Heaviside(phi_T, epsilon=epsilon), gts)


def interpolater(x, s):
    return F.interpolate(x, size=s, mode="bilinear", align_corners=True)


def train(loaders, net, optimizer, epochs=20, T=4, epsilon=-1, dt_max=30):
    for _ in range(epochs):
        totals = 0
        for x, y, dt, sdt in loaders["train"]:
            optimizer.zero_grad()
            phi_0, energy, g = net(x)
            phi_0 = interpolater(phi_0, sdt.shape[-1])
            energy = interpolater(energy, sdt.shape[-1])
            g = torch.sigmoid(interpolater(g, sdt.shape[-1]))
            shift = 10 * np.random.rand() - 5
            phi_T = lse.levelset_evolution(sdt + shift, energy, g, T, dt_max)

            vfs = lse.gradient(y, split=False)
            loss = mean_square_loss(phi_0, sdt) +\
                vector_field_loss(energy, vfs) +\
                LSE_loss(phi_T, y, sdt, epsilon)
            loss.backward()
            clip_grad_norm_(net.parameters(), 10)
            optimizer.step()
            totals += loss.detach().item()
        print(totals)

    # compute validation error
    # save logging information
    return net, optimizer

from utils import lse
import torch
import numpy as np
import torch.nn.functional as F




def validate(net, loader, T, dt_max, epsilon):
    net.eval()
    loss = 0
    for x, y, dt, sdt in loader:
        with torch.no_grad():
            phi_0, energy, g = [interpolater(z, sdt.shape[-1]) for z in net(x)]
            g = torch.sigmoid(g)
            shift = 10 * np.random.rand() - 5
            phi_T = lse.levelset_evolution(phi_0 + shift, energy, g, T, dt_max)
            loss += lse.LSE_loss(phi_T, y, sdt, epsilon).item()

    return loss / len(loader.dataset)


def interpolater(x, s):
    return F.interpolate(x, size=s, mode="bilinear", align_corners=True)

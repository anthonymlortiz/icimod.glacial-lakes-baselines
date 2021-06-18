import torch.nn.functional as F
import torch

def mean_square_loss(y_hat, y, alpha=1):
    assert (y_hat.size() == y.size())
    mse = lambda x: torch.mean(torch.pow(x, 2))
    sdt_loss = mse(y - y_hat)
    return alpha * sdt_loss


def train(loaders, net, optimizer, epochs=20):
    for _ in range(epochs):
        for x, y, sdt in loaders["train"]:
            optimizer.zero_grad()
            phi_0, energy, g = net(x)
            phi_0 = F.interpolate(phi_0, size=sdt.shape[-1], mode='bilinear', align_corners=True)
            #energy = F.interpolate(energy, size=sdt.shape[-1], mode='bilinear', align_corners=True)
            #g = F.sigmoid(F.interpolate(g, size=sdt.shape[-1], mode='bilinear', align_corners=True))

            loss = mean_square_loss(phi_0, sdt)
            loss.backward()
            optimizer.step()

    # compute validation error
    # save logging information
    return net, optimizer

import os
import shutil
import torch
from torch import nn
from torch.nn import functional as F
import math
import matplotlib.pyplot as plt
import pickle


class LocalContextNorm(nn.Module):
    def __init__(self, num_features, channels_per_group=2, window_size=(227, 227), eps=1e-5):
        super(LocalContextNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.channels_per_group = channels_per_group
        self.eps = eps
        self.window_size = window_size

    def forward(self, x):
        N, C, H, W = x.size()
        G = C // self.channels_per_group
        assert C % self.channels_per_group == 0
        if self.window_size[0] < H and self.window_size[1] < W:
            # Build integral image
            device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
            x_squared = x * x
            integral_img = x.cumsum(dim=2).cumsum(dim=3)
            integral_img_sq = x_squared.cumsum(dim=2).cumsum(dim=3)
            # Dilation
            d = (1, self.window_size[0], self.window_size[1])
            integral_img = torch.unsqueeze(integral_img, dim=1)
            integral_img_sq = torch.unsqueeze(integral_img_sq, dim=1)
            kernel = torch.tensor([[[[[1., -1.], [-1., 1.]]]]]).to(device)
            c_kernel = torch.ones((1, 1, self.channels_per_group, 1, 1)).to(device)
            with torch.no_grad():
                # Dilated conv
                sums = F.conv3d(integral_img, kernel, stride=[1, 1, 1], dilation=d)
                sums = F.conv3d(sums, c_kernel, stride=[self.channels_per_group, 1, 1])
                squares = F.conv3d(integral_img_sq, kernel, stride=[1, 1, 1], dilation=d)
                squares = F.conv3d(squares, c_kernel, stride=[self.channels_per_group, 1, 1])
            n = self.window_size[0] * self.window_size[1] * self.channels_per_group
            means = torch.squeeze(sums / n, dim=1)
            var = torch.squeeze((1.0 / n * (squares - sums * sums / n)), dim=1)
            _, _, h, w = means.size()
            pad2d = (int(math.floor((W - w) / 2)), int(math.ceil((W - w) / 2)), int(math.floor((H - h) / 2)),
                     int(math.ceil((H - h) / 2)))
            padded_means = F.pad(means, pad2d, 'replicate')
            padded_vars = F.pad(var, pad2d, 'replicate') + self.eps
            for i in range(G):
                x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] = \
                    (x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] -
                     torch.unsqueeze(padded_means[:, i, :, :], dim=1).to(device)) /\
                    ((torch.unsqueeze(padded_vars[:, i, :, :], dim=1)).to(device)).sqrt()
            del integral_img
            del integral_img_sq
        else:
            x = x.view(N, G, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            x = (x - mean) / (var + self.eps).sqrt()
            x = x.view(N, C, H, W)

        return x * self.weight + self.bias


class CheckpointSaver(object):
    def __init__(self, save_dir, backup_dir):
        self.save_dir = save_dir
        self.backup_dir = backup_dir

    def save(self, state, is_best, checkpoint_name='checkpoint'):
        checkpoint_path = os.path.join(self.save_dir,
                                       '{}.pth.tar'.format(checkpoint_name))
        try:
            shutil.copyfile(
                checkpoint_path,
                '{}_bak'.format(checkpoint_path)
            )
        except IOError:
            pass
        torch.save(state, checkpoint_path)
        if is_best:
            try:
                shutil.copyfile(
                    os.path.join(self.backup_dir,
                                 '{}_best.pth.tar'.format(checkpoint_name)),
                    os.path.join(self.backup_dir,
                                 '{}_best.pth.tar_bak'.format(checkpoint_name))
                )
            except IOError:
                pass
            shutil.copyfile(
                checkpoint_path,
                os.path.join(self.backup_dir,
                             '{}_best.pth.tar'.format(checkpoint_name))
            )


def save_loss(train_loss, val_loss, save_dir, name='loss_plots'):
    """

    :param train_loss: train losses in different epochs
    :param val_loss: validation losses in different epochs
    :return:
    """
    plt.yscale('log')
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper right')
    plt.savefig(save_dir + name + '.png')


# Function to load model options
def load_options(file_name):
    if file_name.endswith(".pkl"):
        opt = pickle.load(open(file_name, 'rb'))
    else:
        opt = pickle.load(open(file_name + '.pkl', 'rb'))
    return opt

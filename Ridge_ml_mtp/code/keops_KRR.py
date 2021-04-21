# based on https://www.kernel-operations.io/keops/_auto_tutorials/interpolation/plot_RBF_interpolation_numpy.html#sphx-glr-auto-tutorials-interpolation-plot-rbf-interpolation-numpy-py
# go check their gallery!

import torch
from matplotlib import pyplot as plt
from pykeops.torch import Vi, Vj
import seaborn as sns
import os

sns.set()
use_cuda = torch.cuda.is_available()
N = 10000 if use_cuda else 1000
seed = 11235813
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
path_file = os.path.dirname(__file__)
save_fig = True


def gaussian_kernel(x, y, sigma=0.2):
    x_i = Vi(x)
    y_j = Vj(y)
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    return (-D_ij / (2 * sigma ** 2)).exp()


if __name__ == "__main__":
    r1, r2 = -5, 5
    x = ((r2 - r1) * torch.rand(N, 1) + r1).type(dtype)
    # signal
    b = x + torch.cos(6*x) * x + 0.1 * torch.randn(N, 1).type(dtype)

    alpha = 1.0  # penalty
    K = gaussian_kernel(x, x)
    a = K.solve(b, alpha=alpha)
    t = torch.linspace(-5, 5, 800).reshape(800, 1).type(dtype)

    K_tx = gaussian_kernel(t, x)
    mean_t = K_tx @ a

    plt.figure()
    plt.scatter(x[:, 0].cpu(), b[:, 0].cpu(), s=.3, label="sample")
    plt.plot(t.cpu(), mean_t.cpu(), "r", label="estimated function")
    # plt.plot(t.cpu(), (t + torch.cos(6*t)*t).cpu(), "orange", label="true function")
    plt.xlabel("x")
    plt.ylabel(r"$\varphi(x)$")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                 "KRR.pdf"))
    plt.show()

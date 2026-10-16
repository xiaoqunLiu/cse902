"""
analysis_utils.py — shared helpers for Problem 2/3/4 analysis (CSE 902 HW1).
"""

import os
import torch
import numpy as np

from vae import VAE
from data import get_loaders

SEED = 1987
LATENT = 16
CONFIGS = ["VAE-A", "VAE-B", "VAE-C", "VAE-D"]
FIG_DIR = "results/figures"


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name, which="final", dev=None):
    dev = dev or device()
    ckpt = torch.load(os.path.join("results", name, f"{which}.pt"), map_location=dev)
    model = VAE(latent_dim=LATENT).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def encode_test_set(model, dev=None):
    """返回整个测试集的 mu, logvar, labels(numpy)。"""
    dev = dev or device()
    _, test_loader = get_loaders(batch_size=256, seed=SEED)
    mus, logvars, labels = [], [], []
    for x, y in test_loader:
        x = x.to(dev)
        mu, logvar = model.encode(x)
        mus.append(mu.cpu()); logvars.append(logvar.cpu()); labels.append(y)
    return (torch.cat(mus).numpy(),
            torch.cat(logvars).numpy(),
            torch.cat(labels).numpy())


@torch.no_grad()
def get_fixed_test_batch(n=64, dev=None):
    """固定取测试集前 n 张图(确定性,不 shuffle),用于重建/干预对比。"""
    dev = dev or device()
    _, test_loader = get_loaders(batch_size=n, seed=SEED)
    x, y = next(iter(test_loader))
    return x[:n].to(dev), y[:n]


def make_grid_np(imgs, nrow):
    """imgs: (N,1,28,28) tensor in [0,1] -> 单张拼接的 numpy (H,W) 用于 imshow。"""
    import torchvision.utils as vutils
    grid = vutils.make_grid(imgs, nrow=nrow, padding=1, pad_value=1.0)
    return grid[0].cpu().numpy()  # 灰度取一个通道


def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)

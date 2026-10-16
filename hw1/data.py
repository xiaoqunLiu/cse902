"""
data.py — Fashion-MNIST loaders for CSE 902 HW1.

像素转到 [0,1](不归一到 [-1,1]),配合 sigmoid 解码 + BCE 重建。
官方 train split 用于训练,官方 test split 只用于评估。
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = "./data"
NUM_CLASSES = 10
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_loaders(batch_size=128, seed=1987, num_workers=4, download=True):
    """返回 (train_loader, test_loader)。shuffle 用种子固定的 generator。"""
    tfm = transforms.ToTensor()  # -> FloatTensor in [0,1], shape (1,28,28)

    train_set = datasets.FashionMNIST(DATA_ROOT, train=True, download=download, transform=tfm)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False, download=download, transform=tfm)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return train_loader, test_loader

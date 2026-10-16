"""
vae.py — Prescribed convolutional VAE for CSE 902 HW1 (Fashion-MNIST).

对应讲解:
  Encoder(压缩)  : 1x28x28 -> 32x14x14 -> 64x7x7 -> flatten(3136) -> [mu(16), logvar(16)]
  Reparameterize  : z = mu + sigma * eps,  sigma = exp(0.5 * logvar),  eps ~ N(0, I)
  Decoder(还原)  : 16 -> 3136 -> reshape 64x7x7 -> 32x14x14 -> 1x28x28 -> sigmoid

架构完全按作业规定:4x4 卷积核, stride 2, padding 1, 隐维度 d_z = 16,
ReLU 隐藏激活, sigmoid 输出 (配合 [0,1] 像素 + BCE 重建损失)。
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # ---------- Encoder:把图越压越小,通道越来越多 ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 1x28x28  -> 32x14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x14x14 -> 64x7x7
            nn.ReLU(inplace=True),
        )
        self.flat_dim = 64 * 7 * 7  # = 3136

        # 两个独立的线性头:一个出 DNA 的"中心" mu,一个出"宽度的对数" logvar
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # ---------- Decoder:从 DNA 反推回图,越放越大 ----------
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)       # 16 -> 3136
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x7x7  -> 32x14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 32x14x14-> 1x28x28
            nn.Sigmoid(),  # 输出像素概率 in (0,1)
        )

    # ---- Encoder:图 -> (mu, logvar) ----
    def encode(self, x: torch.Tensor):
        h = self.encoder(x)                 # N x 64 x 7 x 7
        h = torch.flatten(h, start_dim=1)   # N x 3136
        mu = self.fc_mu(h)                  # N x 16
        logvar = self.fc_logvar(h)          # N x 16
        return mu, logvar

    # ---- 重参数化:把随机性挪到外部噪声 eps 上,整条链路保持可导 ----
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)       # sigma = exp(0.5 * log sigma^2)
        eps = torch.randn_like(std)         # eps ~ N(0, I),与参数无关
        z = mu + std * eps                  # z = mu + sigma ⊙ eps,纯算术,可导
        return z

    # ---- Decoder:z -> 重建图 ----
    def decode(self, z: torch.Tensor):
        h = self.fc_decode(z)               # N x 3136
        h = h.view(-1, 64, 7, 7)            # reshape 回 feature map
        x_hat = self.decoder(h)             # N x 1 x 28 x 28,已过 sigmoid
        return x_hat

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    # ---- 从先验采样生成新图(Problem 2/3 的 prior sampling 用)----
    @torch.no_grad()
    def sample(self, num: int, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(num, self.latent_dim, device=device)  # z ~ N(0, I)
        return self.decode(z)


def count_parameters(model: nn.Module) -> int:
    """实测可训练参数量(报告里用这个数,别用手算估值)。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

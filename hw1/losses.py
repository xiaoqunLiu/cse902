"""
losses.py — beta-VAE 目标函数 + 闭式 KL(CSE 902 HW1)。

损失(负 ELBO 的 beta 版):
    L_beta(x) = 重建项(按像素求和的 BCE)  +  beta * KL(q_phi(z|x) || N(0,I))

对角高斯后验的 KL 闭式解:
    KL = 0.5 * sum_j ( mu_j^2 + exp(logvar_j) - logvar_j - 1 )
"""

import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean"):
    """
    闭式 KL,返回每个样本的 KL(先在隐维度上 sum),再按 batch 规约。

    per_sample_kl[i] = 0.5 * sum_j ( mu_ij^2 + exp(logvar_ij) - logvar_ij - 1 )
    """
    per_sample = 0.5 * torch.sum(
        mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1
    )  # shape: (N,)
    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    elif reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction: {reduction}")


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor, reduction: str = "mean"):
    """
    重建项:逐像素 BCE,先对每张图的所有像素 sum(作业要求 "summed over pixels"),
    再按 batch 规约。x, x_hat 都在 [0,1]。
    """
    # 先不规约,得到逐像素的 BCE,再对像素求和 -> 每张图一个标量
    bce = F.binary_cross_entropy(x_hat, x, reduction="none")  # (N,1,28,28)
    per_sample = bce.flatten(start_dim=1).sum(dim=1)          # (N,) 每张图像素求和
    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    elif reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction: {reduction}")


def beta_vae_loss(x_hat, x, mu, logvar, beta: float = 1.0):
    """
    完整 beta-VAE 损失。返回 (total, recon, kl) 三个标量,
    方便 Problem 2 分别画 recon / KL / total 三条曲线。

    beta 的取值对应四个配置:
        VAE-A: beta=0   VAE-B: beta=1   VAE-C/D: beta=4(D 用 warm-up 动态传入)
    """
    recon = reconstruction_loss(x_hat, x, reduction="mean")
    kl = kl_divergence(mu, logvar, reduction="mean")
    total = recon + beta * kl
    return total, recon, kl


def kl_per_dimension(mu: torch.Tensor, logvar: torch.Tensor):
    """
    逐维 KL(Problem 3-A 诊断后验坍缩用):
        KL_j = (1/2N) * sum_i ( mu_ij^2 + exp(logvar_ij) - logvar_ij - 1 )
    返回 shape (latent_dim,) 的向量,每个元素是该维在整个 batch 上的平均 KL。
    """
    per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)  # (N, D)
    return per_dim.mean(dim=0)  # (D,)

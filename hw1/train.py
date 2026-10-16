"""
train.py — Train the four VAE configurations for CSE 902 HW1.

Configs:
  VAE-A: beta=0, constant
  VAE-B: beta=1, constant
  VAE-C: beta=4, constant
  VAE-D: beta=4, linear warm-up 0 -> 4 over epochs 1-10, then constant 4

记录每个 epoch 的 train/test 的 recon / KL / total,保存:
  results/<name>/metrics.json   逐轮曲线 + 最终指标 + 最佳重建 epoch
  results/<name>/best.pt        测试重建最优的 checkpoint
  results/<name>/final.pt       最后一轮的 checkpoint

用法:
  python train.py --config all        # 跑全部四个(默认)
  python train.py --config B          # 只跑一个
"""

import argparse
import json
import os
import time

import torch

from data import get_loaders
from losses import beta_vae_loss
from vae import VAE, count_parameters

SEED = 1987
EPOCHS = 20
BATCH = 128
LR = 1e-3
LATENT = 16
WARMUP_EPOCHS = 10

CONFIGS = {
    "VAE-A": dict(beta_max=0.0, warmup=False),
    "VAE-B": dict(beta_max=1.0, warmup=False),
    "VAE-C": dict(beta_max=4.0, warmup=False),
    "VAE-D": dict(beta_max=4.0, warmup=True),
}


def beta_at_epoch(epoch, beta_max, warmup):
    """epoch 为 1-indexed。warm-up:epochs 1..10 从 0 线性升到 beta_max。"""
    if not warmup:
        return beta_max
    if epoch >= WARMUP_EPOCHS:
        return beta_max
    return beta_max * (epoch - 1) / (WARMUP_EPOCHS - 1)  # e=1 -> 0, e=10 -> beta_max


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, beta, device):
    model.eval()
    tot_recon = tot_kl = tot_total = 0.0
    n = 0
    for x, _ in loader:
        x = x.to(device)
        x_hat, mu, logvar, _ = model(x)
        total, recon, kl = beta_vae_loss(x_hat, x, mu, logvar, beta=beta)
        bs = x.size(0)
        tot_recon += recon.item() * bs
        tot_kl += kl.item() * bs
        tot_total += total.item() * bs
        n += bs
    return tot_recon / n, tot_kl / n, tot_total / n


def train_config(name, device):
    cfg = CONFIGS[name]
    out_dir = os.path.join("results", name)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(SEED)  # 每个配置从同一个种子出发,保证可比
    train_loader, test_loader = get_loaders(batch_size=BATCH, seed=SEED)

    model = VAE(latent_dim=LATENT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = count_parameters(model)

    print(f"\n=== Training {name} | beta_max={cfg['beta_max']} warmup={cfg['warmup']} "
          f"| params={n_params:,} | device={device} ===")

    history = []
    best_test_recon = float("inf")
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        beta = beta_at_epoch(epoch, cfg["beta_max"], cfg["warmup"])
        model.train()
        t0 = time.time()
        tr_recon = tr_kl = tr_total = 0.0
        n = 0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar, _ = model(x)
            total, recon, kl = beta_vae_loss(x_hat, x, mu, logvar, beta=beta)
            total.backward()
            opt.step()
            bs = x.size(0)
            tr_recon += recon.item() * bs
            tr_kl += kl.item() * bs
            tr_total += total.item() * bs
            n += bs
        tr_recon /= n; tr_kl /= n; tr_total /= n

        te_recon, te_kl, te_total = evaluate(model, test_loader, beta, device)

        history.append(dict(
            epoch=epoch, beta=beta,
            train_recon=tr_recon, train_kl=tr_kl, train_total=tr_total,
            test_recon=te_recon, test_kl=te_kl, test_total=te_total,
        ))

        # 以 test 重建损失为准选最佳 checkpoint
        if te_recon < best_test_recon:
            best_test_recon = te_recon
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": name},
                       os.path.join(out_dir, "best.pt"))

        print(f"  [{name}] epoch {epoch:2d}/{EPOCHS} beta={beta:.3f} "
              f"| train recon={tr_recon:7.2f} kl={tr_kl:6.3f} "
              f"| test recon={te_recon:7.2f} kl={te_kl:6.3f} "
              f"| {time.time()-t0:4.1f}s")

    torch.save({"model": model.state_dict(), "epoch": EPOCHS, "config": name},
               os.path.join(out_dir, "final.pt"))

    metrics = dict(
        config=name, beta_max=cfg["beta_max"], warmup=cfg["warmup"],
        seed=SEED, epochs=EPOCHS, n_params=n_params,
        best_epoch=best_epoch, best_test_recon=best_test_recon,
        final=history[-1], history=history,
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [{name}] done. best test recon={best_test_recon:.2f} @ epoch {best_epoch}. "
          f"final test KL={history[-1]['test_kl']:.3f}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all", choices=["A", "B", "C", "D", "all"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = list(CONFIGS.keys()) if args.config == "all" else [f"VAE-{args.config}"]

    for name in names:
        train_config(name, device)


if __name__ == "__main__":
    main()

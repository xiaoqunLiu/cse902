"""
analyze_p3.py — Problem 3 figures/metrics (CSE 902 HW1): posterior-collapse diagnostics.

产出:
  results/figures/p3_kl_bars.png      四配置逐维 KL 柱状图 + 0.01 阈值线
  results/figures/p3_intervene_<name>.png  原图/重建/先验替换/打乱 代表性网格
  results/p3_summary.json             逐维 KL、sum、active units、Var[mu]、干预 MAE
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis_utils import (CONFIGS, FIG_DIR, LATENT, SEED, device,
                            encode_test_set, ensure_fig_dir,
                            get_fixed_test_batch, load_model, make_grid_np,
                            set_seed)

DEV = device()
ACTIVE_THRESH = 0.01


def per_dim_stats(model):
    mu, logvar, _ = encode_test_set(model, DEV)  # (N,16)
    # KL_j = (1/2N) sum_i (mu^2 + exp(logvar) - logvar - 1)
    kl_j = 0.5 * np.mean(mu**2 + np.exp(logvar) - logvar - 1.0, axis=0)  # (16,)
    var_mu_j = mu.var(axis=0)  # Var_x[mu_j]
    active = int(np.sum(kl_j > ACTIVE_THRESH))
    return kl_j, var_mu_j, active


def kl_bar_plots(stats):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, name in zip(axes.ravel(), CONFIGS):
        kl_j = stats[name]["kl_j"]
        ax.bar(range(LATENT), kl_j, color="#4c72b0")
        ax.axhline(ACTIVE_THRESH, color="red", ls="--", lw=1, label="0.01 threshold")
        ax.set_yscale("log")
        ax.set_title(f"{name}: per-dim KL (sum={sum(kl_j):.2f}, active={stats[name]['active']}/16)")
        ax.set_xlabel("latent dim j"); ax.set_ylabel("KL_j (log)")
        ax.legend(fontsize=7)
    fig.suptitle("Problem 3.A — per-dimension KL with 0.01 active threshold (seed 1987)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/p3_kl_bars.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def interventions(name):
    """返回 MAE(replace/permute vs 普通重建),并存代表性网格。"""
    model = load_model(name, "final", DEV)
    x, _ = get_fixed_test_batch(n=64, dev=DEV)

    set_seed(SEED)
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    recon = model.decode(z)

    z_prior = torch.randn_like(z)              # 独立先验样本
    out_prior = model.decode(z_prior)

    perm = torch.randperm(z.size(0), device=DEV)
    out_perm = model.decode(z[perm])           # 跨样本打乱后验码

    mae_prior = (out_prior - recon).abs().mean().item()
    mae_perm = (out_perm - recon).abs().mean().item()
    # 也报告相对原图的尺度参考
    recon_mae_vs_x = (recon - x).abs().mean().item()

    # 代表性网格:取前 8 张,四行 原图/重建/先验替换/打乱
    k = 8
    rows = torch.cat([x[:k], recon[:k], out_prior[:k], out_perm[:k]], dim=0)
    grid = make_grid_np(rows.cpu(), nrow=k)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(grid, cmap="gray"); ax.axis("off")
    ax.set_title(f"{name}: rows = original / reconstruction / prior-replace / permute "
                 f"(MAE_prior={mae_prior:.3f}, MAE_perm={mae_perm:.3f}, seed 1987)")
    fig.savefig(f"{FIG_DIR}/p3_intervene_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    return dict(mae_prior_replace=mae_prior, mae_permute=mae_perm,
                recon_mae_vs_input=recon_mae_vs_x)


def main():
    ensure_fig_dir()
    stats = {}
    for name in CONFIGS:
        model = load_model(name, "final", DEV)
        kl_j, var_mu_j, active = per_dim_stats(model)
        stats[name] = dict(kl_j=[float(v) for v in kl_j],
                           kl_sum=float(kl_j.sum()),
                           var_mu_j=[float(v) for v in var_mu_j],
                           active=active)
        print(f"{name}: KL sum={kl_j.sum():.3f}, active units={active}/16")

    print("P3: KL bar plots..."); kl_bar_plots(stats)

    interv = {}
    for name in CONFIGS:
        print(f"P3: interventions {name}...")
        interv[name] = interventions(name)
        print(f"   {name}: MAE prior-replace={interv[name]['mae_prior_replace']:.4f}, "
              f"permute={interv[name]['mae_permute']:.4f}")

    # 附最终 recon / KL(来自训练 metrics)便于 C 小节证据表
    finals = {}
    for name in CONFIGS:
        m = json.load(open(f"results/{name}/metrics.json"))
        finals[name] = dict(test_recon=m["final"]["test_recon"],
                            test_kl=m["final"]["test_kl"],
                            best_epoch=m["best_epoch"])

    summary = dict(threshold=ACTIVE_THRESH, per_dim=stats,
                   interventions=interv, finals=finals)
    with open("results/p3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("P3 done.")
    for name in CONFIGS:
        print(f"  {name}: active={stats[name]['active']}/16  KLsum={stats[name]['kl_sum']:.2f}  "
              f"MAE_prior={interv[name]['mae_prior_replace']:.4f}  MAE_perm={interv[name]['mae_permute']:.4f}  "
              f"recon={finals[name]['test_recon']:.2f}")


if __name__ == "__main__":
    main()

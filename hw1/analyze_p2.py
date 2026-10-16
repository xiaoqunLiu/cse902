"""
analyze_p2.py — Problem 2 figures/metrics (CSE 902 HW1).

产出(results/figures/):
  p2_curves.png              四配置 train/test 的 recon / KL / total 曲线
  p2_recon_<name>.png        每配置成对 原图/重建 网格(前 16 张测试图)
  p2_prior_<name>.png        每配置 64 个先验样本(同一组固定先验码)
  p2_pca_<name>.png          每配置 测试隐均值 PCA 二维 + 同一变换下的 N(0,I) 样本
  p2_interp.png              四配置 两张异类图之间 11 点线性插值
  p2_traversal_VAE-B.png     VAE-B 方差最大 3 维在 {-3..3} 上的遍历
并把数值摘要写入 results/p2_summary.json
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from analysis_utils import (CONFIGS, FIG_DIR, SEED, device, encode_test_set,
                            ensure_fig_dir, get_fixed_test_batch, load_model,
                            make_grid_np, set_seed)
from data import CLASS_NAMES

DEV = device()


def plot_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    metric_keys = [("recon", "Reconstruction (BCE/img)"),
                   ("kl", "KL divergence"),
                   ("total", "Total objective")]
    colors = dict(zip(CONFIGS, ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]))
    for ax, (key, title) in zip(axes, metric_keys):
        for name in CONFIGS:
            m = json.load(open(f"results/{name}/metrics.json"))
            ep = [h["epoch"] for h in m["history"]]
            tr = [h[f"train_{key}"] for h in m["history"]]
            te = [h[f"test_{key}"] for h in m["history"]]
            ax.plot(ep, tr, color=colors[name], ls="-", lw=1.4, label=f"{name} train")
            ax.plot(ep, te, color=colors[name], ls="--", lw=1.4, label=f"{name} test")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Problem 2.A — training curves (seed 1987)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/p2_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def recon_grids():
    x, _ = get_fixed_test_batch(n=16, dev=DEV)
    for name in CONFIGS:
        model = load_model(name, "final", DEV)
        x_hat, _, _, _ = model(x)
        # 交错:原图行 + 重建行,拼成 2 组 8 列
        pair = torch.cat([x, x_hat], dim=0)  # 32 张:前16原图,后16重建
        grid = make_grid_np(pair.cpu(), nrow=8)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.imshow(grid, cmap="gray"); ax.axis("off")
        ax.set_title(f"{name}: rows 1-2 original, rows 3-4 reconstruction (test, seed 1987)")
        fig.savefig(f"{FIG_DIR}/p2_recon_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


@torch.no_grad()
def prior_samples():
    # 同一组固定先验码,跨配置复用
    set_seed(SEED)
    z_fixed = torch.randn(64, 16, device=DEV)
    for name in CONFIGS:
        model = load_model(name, "final", DEV)
        imgs = model.decode(z_fixed)
        grid = make_grid_np(imgs.cpu(), nrow=8)
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.imshow(grid, cmap="gray"); ax.axis("off")
        ax.set_title(f"{name}: 64 prior samples z~N(0,I), fixed codes (seed 1987)")
        fig.savefig(f"{FIG_DIR}/p2_prior_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def pca_plots():
    set_seed(SEED)
    summary = {}
    for name in CONFIGS:
        model = load_model(name, "final", DEV)
        mu, _, labels = encode_test_set(model, DEV)
        pca = PCA(n_components=2).fit(mu)
        mu2 = pca.transform(mu)
        prior = np.random.randn(mu.shape[0], 16).astype(np.float32)
        prior2 = pca.transform(prior)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        sc = axes[0].scatter(mu2[:, 0], mu2[:, 1], c=labels, cmap="tab10", s=3, alpha=0.5)
        axes[0].set_title(f"{name}: test latent means (PCA 2-D)")
        cb = fig.colorbar(sc, ax=axes[0], ticks=range(10)); cb.ax.set_yticklabels(CLASS_NAMES, fontsize=6)
        axes[1].scatter(prior2[:, 0], prior2[:, 1], c="gray", s=3, alpha=0.4, label="N(0,I)")
        axes[1].scatter(mu2[:, 0], mu2[:, 1], c="tab:blue", s=3, alpha=0.25, label="posterior means")
        axes[1].set_title(f"{name}: posterior means vs prior (same PCA)")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=0.3)
        fig.suptitle(f"Problem 2.B — aggregate latent distribution ({name}, seed 1987)")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/p2_pca_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        summary[name] = dict(
            explained_var_ratio=[float(v) for v in pca.explained_variance_ratio_],
            posterior_std_pc=[float(mu2[:, 0].std()), float(mu2[:, 1].std())],
            prior_std_pc=[float(prior2[:, 0].std()), float(prior2[:, 1].std())],
        )
    return summary


@torch.no_grad()
def interpolation():
    # 选两张不同类别的测试图:确定性地取测试集里第一张,和第一张与它不同类的图
    x, y = get_fixed_test_batch(n=256, dev=DEV)
    ia = 0
    ib = next(i for i in range(1, len(y)) if y[i] != y[ia])
    alphas = np.linspace(0, 1, 11)

    rows = []
    for name in CONFIGS:
        model = load_model(name, "final", DEV)
        mu, _ = model.encode(x[[ia, ib]])
        za, zb = mu[0], mu[1]
        zs = torch.stack([(1 - a) * za + a * zb for a in alphas], dim=0)
        imgs = model.decode(zs)  # 11 x 1 x 28 x 28
        rows.append(imgs.cpu())
    allrows = torch.cat(rows, dim=0)  # 44 x ...
    grid = make_grid_np(allrows, nrow=11)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.imshow(grid, cmap="gray"); ax.axis("off")
    ax.set_title(f"Problem 2.C — interpolation, rows=A,B,C,D; class {CLASS_NAMES[y[ia]]}→{CLASS_NAMES[y[ib]]} (seed 1987)")
    fig.savefig(f"{FIG_DIR}/p2_interp.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return dict(class_a=CLASS_NAMES[int(y[ia])], class_b=CLASS_NAMES[int(y[ib])])


@torch.no_grad()
def traversal_B():
    name = "VAE-B"
    model = load_model(name, "final", DEV)
    mu, _, _ = encode_test_set(model, DEV)
    var = mu.var(axis=0)  # Var_x[mu_j]
    top3 = np.argsort(var)[::-1][:3]
    vals = [-3, -2, -1, 0, 1, 2, 3]
    base = torch.zeros(16, device=DEV)  # 其他坐标固定为 0

    rows = []
    for d in top3:
        for v in vals:
            z = base.clone(); z[d] = v
            rows.append(model.decode(z.unsqueeze(0)).cpu())
    grid = make_grid_np(torch.cat(rows, 0), nrow=len(vals))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.imshow(grid, cmap="gray"); ax.axis("off")
    ax.set_title(f"{name}: traverse top-3 var dims {list(top3)} over {vals} (seed 1987)")
    fig.savefig(f"{FIG_DIR}/p2_traversal_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return dict(top3_dims=[int(d) for d in top3],
                top3_var=[float(var[d]) for d in top3],
                all_var_sorted=[float(v) for v in np.sort(var)[::-1]])


def main():
    ensure_fig_dir()
    print("P2: curves..."); plot_curves()
    print("P2: recon grids..."); recon_grids()
    print("P2: prior samples..."); prior_samples()
    print("P2: PCA..."); pca_sum = pca_plots()
    print("P2: interpolation..."); interp_sum = interpolation()
    print("P2: traversal (VAE-B)..."); trav_sum = traversal_B()

    summary = dict(pca=pca_sum, interpolation=interp_sum, traversal_B=trav_sum)
    with open("results/p2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("P2 done. figures in", FIG_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
selfcheck_p1.py — Problem 1 的自检脚本(CSE 902 HW1)。

不需要下载数据集,用随机张量就能跑完,验证四件事:
  [1] 逐层 tensor 形状是否符合规定架构
  [2] 可训练参数总数(报告里填实测值)
  [3] 重参数化能否让梯度回流到 encoder(证明可导)
  [4] 闭式 KL 的正确性(边界值 + 与 torch.distributions 对拍)
以及 beta-VAE 损失在 beta=0/1/4 下能否正常前向+反向。

运行:  python selfcheck_p1.py
"""

import torch
import torch.distributions as D

from vae import VAE, count_parameters
from losses import beta_vae_loss, kl_divergence, kl_per_dimension, reconstruction_loss

SEED = 1987  # s = 1000 + 987(来自 README,你的学号末三位 987)


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ok(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    return cond


# ---------------------------------------------------------------- [1] 形状
def check_shapes(model, device):
    print("\n[1] 逐层形状检查 (batch N=8)")
    N = 8
    x = torch.rand(N, 1, 28, 28, device=device)  # [0,1] 像素

    h = model.encoder(x)
    print(f"    输入 x            : {tuple(x.shape)}")
    print(f"    encoder 卷积输出   : {tuple(h.shape)}   (期望 {(N,64,7,7)})")
    mu, logvar = model.encode(x)
    print(f"    mu                : {tuple(mu.shape)}   (期望 {(N,16)})")
    print(f"    logvar            : {tuple(logvar.shape)}   (期望 {(N,16)})")
    z = model.reparameterize(mu, logvar)
    print(f"    z (重参数化后)     : {tuple(z.shape)}   (期望 {(N,16)})")
    x_hat = model.decode(z)
    print(f"    x_hat (重建)       : {tuple(x_hat.shape)}   (期望 {(N,1,28,28)})")

    all_ok = True
    all_ok &= ok(tuple(h.shape) == (N, 64, 7, 7), "encoder 卷积输出 = (N,64,7,7)")
    all_ok &= ok(tuple(mu.shape) == (N, 16) and tuple(logvar.shape) == (N, 16),
                 "mu / logvar = (N,16)")
    all_ok &= ok(tuple(z.shape) == (N, 16), "z = (N,16)")
    all_ok &= ok(tuple(x_hat.shape) == (N, 1, 28, 28), "x_hat = (N,1,28,28)")
    all_ok &= ok(float(x_hat.min()) >= 0.0 and float(x_hat.max()) <= 1.0,
                 "x_hat 在 [0,1](sigmoid 输出)")
    return all_ok


# ---------------------------------------------------------------- [2] 参数量
def check_params(model):
    print("\n[2] 可训练参数量")
    total = count_parameters(model)
    print(f"    合计: {total:,} 个可训练参数")
    print("    逐模块:")
    for name, module in [
        ("Conv 1->32   ", model.encoder[0]),
        ("Conv 32->64  ", model.encoder[2]),
        ("fc_mu        ", model.fc_mu),
        ("fc_logvar    ", model.fc_logvar),
        ("fc_decode    ", model.fc_decode),
        ("ConvT 64->32 ", model.decoder[0]),
        ("ConvT 32->1  ", model.decoder[2]),
    ]:
        n = sum(p.numel() for p in module.parameters())
        print(f"      {name}: {n:,}")
    # 规定架构下应约为 22 万;给个宽松区间防手滑改错结构
    return ok(200_000 <= total <= 240_000, f"参数量落在合理区间 (~220k),实测 {total:,}")


# ---------------------------------------------------------------- [3] 梯度流
def check_grad_flow(model, device):
    print("\n[3] 重参数化的梯度回流检查")
    model.zero_grad()
    x = torch.rand(8, 1, 28, 28, device=device)
    x_hat, mu, logvar, z = model(x)
    loss, recon, kl = beta_vae_loss(x_hat, x, mu, logvar, beta=1.0)
    loss.backward()

    # 关键:encoder 第一层卷积、fc_mu、fc_logvar 都要收到非零梯度,
    # 说明梯度穿过了 z = mu + sigma*eps 这一步(重参数化成功)。
    checks = {
        "encoder Conv1 权重": model.encoder[0].weight.grad,
        "fc_mu 权重": model.fc_mu.weight.grad,
        "fc_logvar 权重": model.fc_logvar.weight.grad,
    }
    all_ok = True
    for name, g in checks.items():
        has_grad = (g is not None) and torch.isfinite(g).all() and (g.abs().sum() > 0)
        all_ok &= ok(bool(has_grad), f"{name} 收到非零有限梯度")
    print(f"    (loss={loss.item():.2f}, recon={recon.item():.2f}, kl={kl.item():.4f})")
    return all_ok


# ---------------------------------------------------------------- [4] KL 正确性
def check_kl(device):
    print("\n[4] 闭式 KL 正确性")
    all_ok = True

    # (a) 边界:mu=0, logvar=0 (即 sigma=1) => 后验=先验 => KL=0
    mu0 = torch.zeros(4, 16, device=device)
    lv0 = torch.zeros(4, 16, device=device)
    kl0 = kl_divergence(mu0, lv0, reduction="mean").item()
    all_ok &= ok(abs(kl0) < 1e-6, f"后验=先验时 KL≈0(实测 {kl0:.2e})")

    # (b) 与 torch.distributions 的 KL 对拍(随机 mu/logvar)
    set_seed()
    mu = torch.randn(32, 16, device=device)
    logvar = torch.randn(32, 16, device=device)
    ours = kl_divergence(mu, logvar, reduction="none")  # (32,)
    std = torch.exp(0.5 * logvar)
    q = D.Normal(mu, std)
    p = D.Normal(torch.zeros_like(mu), torch.ones_like(std))
    ref = D.kl_divergence(q, p).sum(dim=1)              # (32,)
    max_diff = (ours - ref).abs().max().item()
    all_ok &= ok(max_diff < 1e-4, f"与 torch.distributions 对拍最大误差 {max_diff:.2e}")

    # (c) 逐维 KL 形状(Problem 3 会用)
    per_dim = kl_per_dimension(mu, logvar)
    all_ok &= ok(tuple(per_dim.shape) == (16,), f"逐维 KL 形状 = (16,),和={per_dim.sum():.3f}")
    return all_ok


# ---------------------------------------------------------------- [5] beta 扫一遍
def check_beta_variants(model, device):
    print("\n[5] beta-VAE 损失在 beta=0/1/4 下前向+反向")
    all_ok = True
    x = torch.rand(8, 1, 28, 28, device=device)
    for beta in (0.0, 1.0, 4.0):
        model.zero_grad()
        x_hat, mu, logvar, z = model(x)
        total, recon, kl = beta_vae_loss(x_hat, x, mu, logvar, beta=beta)
        total.backward()
        finite = torch.isfinite(total)
        print(f"    beta={beta:<3}  total={total.item():8.2f}  recon={recon.item():8.2f}  kl={kl.item():7.4f}")
        all_ok &= ok(bool(finite), f"beta={beta} 损失有限且可反向")
    return all_ok


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== CSE 902 HW1 — Problem 1 自检 ===")
    print(f"seed = {SEED}   device = {device}")

    model = VAE(latent_dim=16).to(device)

    results = [
        check_shapes(model, device),
        check_params(model),
        check_grad_flow(model, device),
        check_kl(device),
        check_beta_variants(model, device),
    ]
    print("\n==================================================")
    if all(results):
        print("全部自检通过 ✓  —— 模型 / 重参数化 / β-VAE 损失 / 闭式 KL 均正确")
    else:
        print("有自检未通过 ✗  —— 请检查上面标 [FAIL] 的项")


if __name__ == "__main__":
    main()

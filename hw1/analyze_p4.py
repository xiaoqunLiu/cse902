"""
analyze_p4.py — Problem 4 consolidated comparison table (CSE 902 HW1).

汇总 P2/P3 的数值证据,写出:
  results/p4_comparison.csv
  results/p4_comparison.json
供 Problem 4.A 的多准则对比表使用。
"""

import csv
import json

from analysis_utils import CONFIGS


def main():
    p2 = json.load(open("results/p2_summary.json"))
    p3 = json.load(open("results/p3_summary.json"))

    rows = []
    for name in CONFIGS:
        fin = p3["finals"][name]
        pca = p2["pca"][name]
        interv = p3["interventions"][name]
        per = p3["per_dim"][name]
        # 聚集后验-先验尺度匹配:posterior PC std / prior PC std 的均值(越接近 1 越匹配)
        post = pca["posterior_std_pc"]; prior = pca["prior_std_pc"]
        scale_ratio = 0.5 * (post[0] / prior[0] + post[1] / prior[1])
        rows.append(dict(
            config=name,
            test_recon=round(fin["test_recon"], 2),
            test_kl_sum=round(fin["test_kl"], 3),
            active_units=per["active"],
            pca_scale_ratio=round(scale_ratio, 2),  # ~1 好;>>1 后验远大于先验
            mae_prior_replace=round(interv["mae_prior_replace"], 4),
            mae_permute=round(interv["mae_permute"], 4),
            best_recon_epoch=fin["best_epoch"],
        ))

    with open("results/p4_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open("results/p4_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 打印成表
    cols = list(rows[0].keys())
    print(" | ".join(f"{c:>16s}" for c in cols))
    print("-" * (len(cols) * 19))
    for r in rows:
        print(" | ".join(f"{str(r[c]):>16s}" for c in cols))


if __name__ == "__main__":
    main()

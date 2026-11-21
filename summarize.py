#!/usr/bin/env python
"""Cross-subject summary tables with BH-FDR correction across ROI x layer per subject.

Usage: /scratch/arnav.shakya/conda/envs/rsa/bin/python summarize.py sub-01 sub-03 [sub-02 sub-06]
"""
import os
import sys
import numpy as np
import pandas as pd

RES = os.environ.get("RES_OUT", "/scratch/arnav.shakya/rsa_work/results")
subs = sys.argv[1:] or ["sub-01", "sub-02", "sub-03", "sub-06"]
subs = [s for s in subs if os.path.exists(f"{RES}/{s}_rsa_overall.csv")]


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


# 1. Overall alignment at best layer per subject x ROI x model
ov_list = []
for s in subs:
    df = pd.read_csv(f"{RES}/{s}_rsa_overall.csv")
    df["subject"] = s
    ov_list.append(df)
ov = pd.concat(ov_list, ignore_index=True)

print("======================================================================")
print(f"1. OVERALL ALIGNMENT (best layer per subject x ROI x model) | Subjects: {subs}")
print("======================================================================")
best_rows = []
for (s, r, m), g in ov.groupby(["subject", "roi", "model"], sort=False):
    top = g.loc[g.rho.idxmax()]
    best_rows.append(top)
best_df = pd.DataFrame(best_rows)
piv_ov = best_df.pivot(index=["roi", "model"], columns="subject", values="rho")
if len(subs) > 1:
    piv_ov["mean_rho"] = piv_ov[subs].mean(axis=1)
print(piv_ov.round(3).to_string())

# 2. Clutter contrast (low_vs_high) at best overall layer + FDR across ROI x layer
cl_list = []
for s in subs:
    df = pd.read_csv(f"{RES}/{s}_rsa_clutter.csv")
    df["subject"] = s
    df["q_fdr"] = np.nan
    for (c, m), g in df.groupby(["contrast", "model"]):
        df.loc[g.index, "q_fdr"] = bh_fdr(g["delta_p_boot"].values)
    cl_list.append(df)
cl = pd.concat(cl_list, ignore_index=True)
cl.to_csv(f"{RES}/summary_clutter_fdr.csv", index=False)

print("\n======================================================================")
print("2. CLUTTER CONTRAST (low_vs_high): delta = rho_high - rho_low at best overall layer")
print("   (* p < 0.05 uncorrected, ** q_FDR < 0.05 across ROI x layer)")
print("======================================================================")
lvh = cl[cl["contrast"] == "low_vs_high"].merge(
    best_df[["subject", "roi", "model", "layer"]], on=["subject", "roi", "model", "layer"]
)
for _, row in lvh.iterrows():
    sig = "**" if row["q_fdr"] < 0.05 else ("*" if row["delta_p_boot"] < 0.05 else "")
    print(f"{row['subject']:6s} {row['roi']:4s} {row['model']:10s} {row['layer']:7s} "
          f"low={row['rho_low']:6.3f} high={row['rho_high']:6.3f} "
          f"delta={row['delta_high_minus_low']:+6.3f} [{row['delta_ci_lo']:+6.3f},{row['delta_ci_hi']:+6.3f}] "
          f"p={row['delta_p_boot']:.3f} q={row['q_fdr']:.3f} {sig}")

# 3. Supervised vs SimCLR difference at layer3 & layer4
md_list = []
for s in subs:
    df = pd.read_csv(f"{RES}/{s}_rsa_model_diff.csv")
    df["subject"] = s
    df["q_fdr"] = np.nan
    for a, g in df.groupby("analysis"):
        df.loc[g.index, "q_fdr"] = bh_fdr(g["p_boot"].values)
    md_list.append(df)
md = pd.concat(md_list, ignore_index=True)
md.to_csv(f"{RES}/summary_model_diff_fdr.csv", index=False)

print("\n======================================================================")
print("3. SUPERVISED - SIMCLR (overall analysis, layers 1-4 & avgpool)")
print("======================================================================")
md_ov = md[md["analysis"] == "overall"]
piv_md = md_ov.pivot(index=["roi", "layer"], columns="subject", values="supervised_minus_simclr")
if len(subs) > 1:
    piv_md["mean_diff"] = piv_md[subs].mean(axis=1)
print(piv_md.round(3).to_string())

#!/usr/bin/env python
"""Generate publication-ready figures for CNeuroMod-THINGS RSA project.

Usage: python plot_figures.py [sub-01 sub-03 sub-06 sub-02]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RES = os.environ.get("RES_OUT", "/scratch/arnav.shakya/rsa_work/results")
OUT_FIG = os.environ.get("FIG_OUT", "/scratch/arnav.shakya/rsa_work/figures")
os.makedirs(OUT_FIG, exist_ok=True)

subs = sys.argv[1:] or ["sub-01", "sub-03", "sub-06"]
subs = [s for s in subs if os.path.exists(f"{RES}/{s}_rsa_overall.csv")]

# Colors for models
MODEL_COLORS = {
    "supervised": "#2b5c8f",  # Deep Blue
    "simclr": "#e07a5f",      # Terracotta / Coral
    "random": "#81b29a"       # Muted Sage Green
}

def plot_overall():
    dfs = []
    for s in subs:
        df = pd.read_csv(f"{RES}/{s}_rsa_overall.csv")
        df["subject"] = s
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    
    # Get best layer per ROI x model x subject
    best = []
    for (s, r, m), g in data.groupby(["subject", "roi", "model"]):
        best.append(g.loc[g.rho.idxmax()])
    best_df = pd.DataFrame(best)
    
    # Average across subjects
    summary = best_df.groupby(["roi", "model"]).agg(
        mean_rho=("rho", "mean"),
        sem_rho=("rho", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=summary, x="roi", y="mean_rho", hue="model", 
        palette=MODEL_COLORS, edgecolor="black", linewidth=0.8, ax=ax
    )
    ax.set_title("Overall Category-Level RSA Alignment Across ROIs", pad=15, fontweight="bold")
    ax.set_xlabel("Brain Region of Interest (ROI)")
    ax.set_ylabel(r"Spearman's $\rho$ (Model vs. Brain RDM)")
    ax.legend(title="DNN Model", frameon=True, framealpha=0.9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_path = f"{OUT_FIG}/fig1_overall_alignment.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"))
    print(f"Saved {out_path}")
    plt.close(fig)

def plot_clutter():
    dfs = []
    for s in subs:
        p = f"{RES}/{s}_rsa_clutter.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["subject"] = s
            dfs.append(df)
    if not dfs:
        print("No clutter CSVs found, skipping figure 2.")
        return
    data = pd.concat(dfs, ignore_index=True)
    lvh = data[data["contrast"] == "low_vs_high"]
    
    # Filter for supervised model at best layer
    summary = lvh[lvh["model"] == "supervised"].groupby("roi").agg(
        mean_delta=("delta_high_minus_low", "mean"),
        sem_delta=("delta_high_minus_low", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e07a5f" if d < 0 else "#2b5c8f" for d in summary["mean_delta"]]
    ax.bar(summary["roi"], summary["mean_delta"], yerr=summary["sem_delta"], 
           color=colors, edgecolor="black", linewidth=0.8, capsize=4)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Effect of Scene Clutter on Brain-DNN Alignment (High vs. Low Clutter)", pad=15, fontweight="bold")
    ax.set_xlabel("Brain Region of Interest (ROI)")
    ax.set_ylabel(r"$\Delta \rho$ (High Clutter − Low Clutter)")
    plt.tight_layout()
    out_path = f"{OUT_FIG}/fig2_clutter_contrast.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"))
    print(f"Saved {out_path}")
    plt.close(fig)

def plot_layer_profiles():
    dfs = []
    for s in subs:
        df = pd.read_csv(f"{RES}/{s}_rsa_overall.csv")
        df["subject"] = s
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    
    summary = data.groupby(["roi", "model", "layer"]).agg(
        mean_rho=("rho", "mean")
    ).reset_index()
    
    layer_order = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
    rois = [r for r in ["V1", "V2", "V3", "EBA", "FFA", "OFA", "PPA", "OPA", "MPA"] if r in summary["roi"].unique()]
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10), sharey=True)
    axes = axes.flatten()
    
    for i, roi in enumerate(rois):
        ax = axes[i]
        sub_df = summary[summary["roi"] == roi]
        sns.lineplot(
            data=sub_df, x="layer", y="mean_rho", hue="model", 
            palette=MODEL_COLORS, marker="o", ax=ax, errorbar=None, linewidth=2
        )
        ax.set_title(roi, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(r"$\rho$" if i % 3 == 0 else "")
        ax.set_xticklabels(["L1", "L2", "L3", "L4", "Pool"], rotation=0)
        if i > 0:
            ax.get_legend().remove()
        else:
            ax.legend(frameon=True, framealpha=0.9, fontsize=8)
            
    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle("Layer-wise Representational Alignment Across Visual Hierarchy", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = f"{OUT_FIG}/fig3_layer_profiles.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"))
    print(f"Saved {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    print(f"Generating figures from {RES} for subjects: {subs}")
    plot_overall()
    plot_clutter()
    plot_layer_profiles()
    print(f"All figures successfully saved to {OUT_FIG}/")

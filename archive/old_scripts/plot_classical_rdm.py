#!/usr/bin/env python
"""Generate classical RSA Representational Dissimilarity Matrix (RDM) figures.

Usage: python plot_classical_rdm.py [sub-01 sub-03 sub-06 sub-02]
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

BRAIN = os.environ.get("RSA_OUT", "/scratch/arnav.shakya/rsa_work/brain")
OUT_FIG = os.environ.get("FIG_OUT", "/scratch/arnav.shakya/rsa_work/figures")
os.makedirs(OUT_FIG, exist_ok=True)

subs = sys.argv[1:] or ["sub-01", "sub-02", "sub-03", "sub-06"]

def plot_rdms(sub):
    rdm_path = f"{BRAIN}/{sub}_category_rdms.npz"
    if not os.path.exists(rdm_path):
        print(f"RDM file not found for {sub}: {rdm_path}")
        return
        
    data = np.load(rdm_path)
    rdm_keys = [k for k in data.files if k.startswith("rdm_")]
    
    if not rdm_keys:
        print(f"No ROI RDMs found in {rdm_path}")
        return

    n_rdms = len(rdm_keys)
    cols = 3
    rows = (n_rdms + cols - 1) // cols
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 3.5))
    axes = np.atleast_1d(axes).flatten()
    
    for i, key in enumerate(rdm_keys):
        ax = axes[i]
        roi_name = key.replace("rdm_", "").upper()
        
        # Directly load the square (n_cat, n_cat) dissimilarity matrix
        rdm_mat = data[key]
        
        im = ax.imshow(rdm_mat, cmap="viridis", aspect="auto", origin="upper")
        ax.set_title(f"{roi_name} RDM", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle(f"Classical Category-Level RDMs Across ROIs ({sub})", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_path = f"{OUT_FIG}/{sub}_classical_rdms.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"))
    print(f"Saved classical RDM figure to {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    for s in subs:
        print(f"Plotting classical RDMs for {s}...")
        plot_rdms(s)

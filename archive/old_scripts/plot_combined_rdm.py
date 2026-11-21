#!/usr/bin/env python
"""Generate Group Human vs. DNN Model RDM comparison figures.

Usage: python plot_combined_rdm.py [sub-01 sub-02 sub-03 sub-06]
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
FEAT = os.environ.get("FEAT_OUT", "/scratch/arnav.shakya/rsa_work/features")
OUT_FIG = os.environ.get("FIG_OUT", "/scratch/arnav.shakya/rsa_work/figures")
os.makedirs(OUT_FIG, exist_ok=True)

subs = sys.argv[1:] or ["sub-01", "sub-02", "sub-03", "sub-06"]

def compute_dist_matrix(X):
    """Correlation distance between rows after mean-centering."""
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    X = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-8
    X = X / norm
    return (1.0 - X @ X.T).astype(np.float32)

def vector_to_square(vec):
    n_cat = int((1 + np.sqrt(1 + 8 * len(vec))) / 2)
    mat = np.zeros((n_cat, n_cat), dtype=np.float32)
    iu = np.triu_indices(n_cat, 1)
    mat[iu] = vec
    return mat + mat.T

def main():
    print(f"Loading human RDMs for subjects: {subs}")
    
    # 1. Collect and average human RDMs across subjects
    roi_rdms_collected = {}
    for s in subs:
        path = f"{BRAIN}/{s}_category_rdms.npz"
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {s}.")
            continue
        d = np.load(path)
        for k in d.files:
            if k.startswith("rdm_"):
                roi_name = k.replace("rdm_", "")
                if roi_name not in roi_rdms_collected:
                    roi_rdms_collected[roi_name] = []
                roi_rdms_collected[roi_name].append(d[k])
                
    if not roi_rdms_collected:
        print("Error: No human ROI RDMs found across specified subjects.")
        sys.exit(1)
        
    group_human_rdms = {
        roi: np.mean(vecs, axis=0) for roi, vecs in roi_rdms_collected.items()
    }

    # 2. Load Model features and compute Model RDMs
    print("Computing model RDMs from feature files...")
    model_rdms = {}
    models_to_plot = [
        ("supervised", "layer1", "Sup. Layer 1"),
        ("supervised", "layer3", "Sup. Layer 3"),
        ("supervised", "avgpool", "Sup. Avgpool"),
        ("simclr", "layer3", "SimCLR Layer 3"),
        ("random", "layer1", "Random Layer 1")
    ]
    
    for m_name, layer_name, label in models_to_plot:
        feat_path = f"{FEAT}/{m_name}_features.npz"
        if os.path.exists(feat_path):
            f_data = np.load(feat_path)
            if layer_name in f_data:
                X_feat = f_data[layer_name]
                # If too large or to keep visualization clean, we can take a subset of categories or use full
                D = compute_dist_matrix(X_feat)
                iu = np.triu_indices(D.shape[0], 1)
                model_rdms[label] = D[iu]
            else:
                print(f"Layer {layer_name} not found in {feat_path}")
        else:
            print(f"Feature file not found: {feat_path}")

    # 3. Plotting: Group Human ROIs vs Model Layers side-by-side
    target_human_rois = [r for r in ["v1", "v2", "ffa", "ppa", "eba", "ofa"] if r in group_human_rdms]
    n_human = len(target_human_rois)
    n_models = len(model_rdms)
    
    total_plots = n_human + n_models
    cols = 4
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 3.5, rows * 3.2))
    axes = np.atleast_1d(axes).flatten()
    
    plot_idx = 0
    # Plot Group Human ROIs
    for roi in target_human_rois:
        ax = axes[plot_idx]
        mat = vector_to_square(group_human_rdms[roi])
        im = ax.imshow(mat, cmap="viridis", aspect="auto", origin="upper")
        ax.set_title(f"Human: {roi.upper()}", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plot_idx += 1
        
    # Plot Model Layers
    for label, rdm_vec in model_rdms.items():
        if plot_idx >= len(axes):
            break
        ax = axes[plot_idx]
        mat = vector_to_square(rdm_vec)
        im = ax.imshow(mat, cmap="viridis", aspect="auto", origin="upper")
        ax.set_title(f"DNN: {label}", fontweight="bold", color="#2b5c8f")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plot_idx += 1
        
    # Hide any unused subplots
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle("Group-Averaged Human Brain RDMs vs. DNN Model RDMs", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_path = f"{OUT_FIG}/group_human_vs_models_rdms.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"))
    print(f"Saved combined RDM comparison figure to {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()

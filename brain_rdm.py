#!/usr/bin/env python
"""Brain RDMs and sanity checks from extracted ROI betas (no stimulus images needed).

Usage: python brain_rdm.py sub-01 [sub-02 ...]
Reads : RSA_OUT/<sub>_roi_betas.npz, <sub>_roi_info.json, the annotations json
Writes: RSA_OUT/<sub>_category_rdms.npz  (category-level RDMs per ROI, for later model comparison)
Prints: per ROI - category index, animacy index, exemplar split-half reliability, mean noise ceiling
"""
import json
import os
import re
import sys
import traceback

import nibabel as nib
import numpy as np
from scipy.stats import spearmanr

ROOT = os.environ.get("THINGS_ROOT", "/scratch/arnav.shakya/cneuromod-things")
BRAIN = os.environ.get("RSA_OUT", "/scratch/arnav.shakya/rsa_work/brain")
GS = f"{ROOT}/THINGS/glmsingle"
ANN = f"{GS}/task-things_imgAnnotations.json"
MIN_REPS = 3  # primary image set: images with all 3 repetitions (most reliable betas)
PAPER_MAX_NC = {"sub-01": 56.51, "sub-02": 66.43, "sub-03": 73.89}  # maximal voxel noise ceilings in the dataset paper


def corr_dist(X):
    """Correlation distance between rows (conditions) after removing each voxel's mean across conditions."""
    return 1.0 - np.corrcoef(X - X.mean(0, keepdims=True))


def upper(D):
    return D[np.triu_indices(D.shape[0], 1)]


def exemplar_number(name):
    m = re.search(r"_(\d+)[a-z]?$", name)
    return int(m.group(1)) if m else -1


def category_patterns(X, cats, keep_cats):
    return np.stack([X[cats == c].mean(0) for c in keep_cats])


def find_mask(sub, fname):
    for p in (f"{ROOT}/fLoc/rois/{sub}/rois/task-derived/{fname}",
              f"{ROOT}/retinotopy/prf/{sub}/rois/{fname}"):
        if os.path.exists(p):
            return p
    return None


def run(sub):
    z = np.load(f"{BRAIN}/{sub}_roi_betas.npz")
    info = json.load(open(f"{BRAIN}/{sub}_roi_info.json"))
    ann = json.load(open(ANN))
    rois = [r for r in info if r in z.files]

    names = np.array([str(n) for n in z["images"]])
    keep = z["num_reps"] >= MIN_REPS
    names = names[keep]
    cats = np.array([ann[n]["image_category"] for n in names])
    ex = np.array([exemplar_number(n) for n in names])
    living = np.array([np.nan if ann[n].get("categ_living") is None else float(ann[n]["categ_living"])
                       for n in names])
    is_living = living > 4.0
    valid_liv = ~np.isnan(living)
    print(f"[{sub}] images with >= {MIN_REPS} reps: {len(names)} | categories: {len(set(cats))} "
          f"| living: {int((is_living & valid_liv).sum())} non-living: {int((~is_living & valid_liv).sum())}")

    n = len(names)
    iu = np.triu_indices(n, 1)
    same_cat = (cats[:, None] == cats[None, :])[iu]
    same_liv = (is_living[:, None] == is_living[None, :])[iu]
    both_valid = (valid_liv[:, None] & valid_liv[None, :])[iu]

    nc = None
    ncp = f"{GS}/{sub}/glmsingle/output/{sub}_task-things_space-T1w_model-fitHrfGLMdenoiseRR_stat-noiseCeilings_statmap.nii.gz"
    if os.path.exists(ncp):
        nc = nib.load(ncp).get_fdata()
        ref = f" (paper: {PAPER_MAX_NC[sub]})" if sub in PAPER_MAX_NC else ""
        print(f"[{sub}] noise-ceiling map found, max {np.nanmax(nc):.2f}{ref}")

    all_cats = np.array(sorted(set(cats)))
    odd, even = (ex % 2 == 1), (ex % 2 == 0)
    cats_a, cats_b = set(cats[odd]), set(cats[even])
    both = np.array(sorted(cats_a & cats_b))
    out, rows = {"categories": all_cats, "categories_split_half": both}, []

    for roi in rois:
        X = z[roi][keep].astype(np.float64)
        D = corr_dist(X)
        d = D[iu]
        cat_index = d[~same_cat].mean() - d[same_cat].mean()
        m = ~same_cat & both_valid
        liv_index = d[m & ~same_liv].mean() - d[m & same_liv].mean()

        out[f"rdm_{roi}"] = corr_dist(category_patterns(X, cats, all_cats)).astype(np.float32)
        ra = upper(corr_dist(category_patterns(X[odd], cats[odd], both)))
        rb = upper(corr_dist(category_patterns(X[even], cats[even], both)))
        r = spearmanr(ra, rb)[0]
        sb = 2 * r / (1 + r)

        nc_mean = np.nan
        if nc is not None:
            mp = find_mask(sub, info[roi]["file"])
            if mp is not None:
                mk = nib.load(mp).get_fdata() > 0.5
                if mk.shape == nc.shape:
                    nc_mean = float(np.nanmean(nc[mk]))
        rows.append((roi, X.shape[1], cat_index, liv_index, r, sb, nc_mean))

    np.savez_compressed(f"{BRAIN}/{sub}_category_rdms.npz", **out)
    print(f"[{sub}] category index  = mean distance between categories - within category (higher = more category info)")
    print(f"[{sub}] animacy index   = living vs non-living distance - within-class distance (different categories only)")
    print(f"[{sub}] split-half r    = Spearman between category RDMs built from odd vs even exemplars "
          f"({len(both)} categories); SB = Spearman-Brown corrected")
    print(f"{'ROI':6s}{'nvox':>6s}{'cat_idx':>10s}{'anim_idx':>10s}{'split_r':>9s}{'SB':>7s}{'meanNC%':>9s}")
    for roi, nv, ci, li, r, sb, ncm in rows:
        print(f"{roi:6s}{nv:6d}{ci:10.4f}{li:10.4f}{r:9.3f}{sb:7.3f}{ncm:9.2f}")
    print(f"[{sub}] saved {BRAIN}/{sub}_category_rdms.npz")


if __name__ == "__main__":
    failed = []
    for s in sys.argv[1:]:
        try:
            run(s)
        except Exception:
            traceback.print_exc()
            failed.append(s)
    if failed:
        print("FAILED:", failed)
        sys.exit(1)

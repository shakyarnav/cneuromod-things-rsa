#!/usr/bin/env python
"""Extract ROI patterns from CNeuroMod-THINGS image-wise GLMsingle betas (T1w space).

Usage: python extract_roi_betas.py sub-01 sub-02 ...
Output: OUT/<sub>_roi_betas.npz  (images, num_reps, one images x voxels array per ROI)
        OUT/<sub>_roi_info.json  (mask file and voxel counts per ROI)
"""
import glob
import json
import os
import sys
import traceback

import h5py
import nibabel as nib
import numpy as np

ROOT = os.environ.get("THINGS_ROOT", "/scratch/arnav.shakya/cneuromod-things")
OUT = os.environ.get("RSA_OUT", "/scratch/arnav.shakya/rsa_work/brain")
GS = f"{ROOT}/THINGS/glmsingle"
FLOC = f"{ROOT}/fLoc/rois"
RETINO = f"{ROOT}/retinotopy/prf"

EARLY = ["V1", "V2", "V3"]
HIGH = ["FFA", "OFA", "PPA", "OPA", "MPA", "EBA"]


def pick(cands, what):
    cands = sorted(cands)
    if len(cands) > 1:  # prefer the tight (unsmoothed) mask used in the CNeuroMod-THINGS paper
        unsmooth = [c for c in cands if "desc-unsmooth" in c]
        if len(unsmooth) == 1:
            cands = unsmooth
    if len(cands) != 1:
        raise RuntimeError(f"{what}: expected exactly 1 mask, found {len(cands)}: "
                           f"{[os.path.basename(c) for c in cands]}")
    if not os.path.exists(cands[0]):  # broken symlink = annexed file not downloaded yet
        raise FileNotFoundError(f"{what}: not downloaded yet (datalad get it): {cands[0]}")
    return cands[0]


def run(sub):
    h5p = (f"{GS}/{sub}/glmsingle/output/{sub}_task-things_space-T1w_"
           "model-fitHrfGLMdenoiseRR_stat-imageBetas_desc-zscore_statseries.h5")
    brp = f"{GS}/{sub}/glmsingle/input/{sub}_task-things_space-T1w_label-brain_desc-unionNonNaN_mask.nii"
    for p in (h5p, brp):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing (datalad get it first): {p}")

    brain = nib.load(brp).get_fdata() > 0
    nvec = int(brain.sum())
    lut = np.full(brain.size, -1, dtype=np.int64)      # grid voxel (C-order) -> position in beta vector
    lut[np.flatnonzero(brain.ravel())] = np.arange(nvec)

    mask_files = {}
    for roi in HIGH:
        mask_files[roi] = pick(glob.glob(f"{FLOC}/{sub}/rois/task-derived/*_roi-{roi}_*mask.nii.gz"),
                               f"{sub} fLoc {roi}")
    if os.path.isdir(f"{RETINO}/{sub}"):
        for roi in EARLY:
            mask_files[roi] = pick(glob.glob(f"{RETINO}/{sub}/rois/*res-func*label-{roi}_desc-nn_mask.nii.gz"),
                                   f"{sub} retinotopy {roi}")
    else:
        print(f"[{sub}] no retinotopy data: skipping {EARLY}")

    idx, info = {}, {}
    for roi, path in mask_files.items():
        m = nib.load(path).get_fdata() > 0.5
        if m.shape != brain.shape:
            raise RuntimeError(f"{sub} {roi}: mask grid {m.shape} != beta grid {brain.shape}")
        vox = np.flatnonzero((m & brain).ravel())
        idx[roi] = lut[vox]
        info[roi] = dict(file=os.path.basename(path), n_mask=int(m.sum()), n_used=int(len(vox)))
        if len(vox) < 0.9 * m.sum():
            print(f"[{sub}] WARNING {roi}: only {len(vox)}/{int(m.sum())} mask voxels are inside the beta mask")

    with h5py.File(h5p, "r") as h:
        if "mask_array" in h:
            ma = np.asarray(h["mask_array"][()])
            if ma.shape == brain.shape:
                if not np.array_equal(ma > 0, brain):
                    raise RuntimeError(f"{sub}: h5 mask_array differs from the unionNonNaN mask")
                print(f"[{sub}] h5 mask_array matches the unionNonNaN mask ({nvec} voxels)")
            else:
                print(f"[{sub}] WARNING mask_array shape {ma.shape} != {brain.shape}; using the nifti mask")
        names = sorted(k for k in h.keys() if isinstance(h[k], h5py.Group) and "betas" in h[k])
        reps = np.array([int(np.ravel(h[k]["num_reps"][()])[0]) for k in names])
        out = {roi: np.empty((len(names), len(ix)), np.float32) for roi, ix in idx.items()}
        for i, k in enumerate(names):
            v = h[k]["betas"][()]
            if v.shape[0] != nvec:
                raise RuntimeError(f"{sub} {k}: beta vector {v.shape[0]} != mask voxels {nvec}")
            for roi, ix in idx.items():
                out[roi][i] = v[ix]
            if (i + 1) % 1000 == 0:
                print(f"[{sub}] {i + 1}/{len(names)} images", flush=True)

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/{sub}_roi_betas.npz", images=np.array(names), num_reps=reps, **out)
    json.dump(info, open(f"{OUT}/{sub}_roi_info.json", "w"), indent=1)
    print(f"[{sub}] images: {len(names)} | reps: { {int(r): int((reps == r).sum()) for r in np.unique(reps)} }")
    print(f"[{sub}] voxels per ROI: { {r: info[r]['n_used'] for r in info} }")


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

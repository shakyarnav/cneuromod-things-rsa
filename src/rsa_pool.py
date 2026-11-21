#!/usr/bin/env python
"""Category-level RSA over all multi-resolution readouts.
Usage: python rsa_pool.py sub-01 [sub-02 ...]      env: MODELS, N_BOOT (1000)
Writes RES/<sub>_rsa_pooling.csv        (roi, model, readout, stage, block, grid, rho, CI, SB, ceiling)
       RES/<sub>_rsa_pooling_diff.csv   (pairwise model differences with bootstrap CI and p)
"""
import json, os, re, sys, traceback
import numpy as np
import pandas as pd
from rsa import (BRAIN, FEAT, RES, ANN, MIN_REPS, SEED, N_BOOT, dist_matrix, zrank, ci_p,
                 split_half_reliability, exemplar_number)

MULTI = f"{FEAT}/multi"
MODELS = os.environ.get("MODELS", "supervised,simclr,random").split(",")
PAIRS = [("supervised", "simclr"), ("supervised", "random"), ("simclr", "random")]


def parse(key):
    if key == "avgpool":
        return "avgpool", 0, 1
    m = re.match(r"(stem|layer\d)(?:_b(\d+))?_g(\d+)$", key)
    return m.group(1), int(m.group(2) or 0), int(m.group(3))


def fast_boot(Rb, Rm, ci0, ci1, ks):
    Rb2, Rm2 = Rb ** 2, Rm ** 2
    def corr(w):
        W = w.sum()
        mb, mm = Rb @ w / W, Rm @ w / W
        cov = (Rb * w) @ Rm.T / W - np.outer(mb, mm)
        vb, vm = Rb2 @ w / W - mb ** 2, Rm2 @ w / W - mm ** 2
        return cov / np.sqrt(np.outer(vb, vm))
    est = corr(np.ones(len(ci0)))
    boots = np.empty((len(ks),) + est.shape)
    for b, k in enumerate(ks):
        boots[b] = corr(k[ci0] * k[ci1])
    return est, boots


def run(sub):
    rng = np.random.default_rng(SEED)
    z = np.load(f"{BRAIN}/{sub}_roi_betas.npz")
    ann = json.load(open(ANN))
    rois = [k for k in z.files if k not in ("images", "num_reps")]
    keep = z["num_reps"] >= MIN_REPS
    names = [str(x) for x in z["images"][keep]]
    n = len(names)
    cats_all = np.array([ann[x]["image_category"] for x in names])
    cat_list, cat_of = np.unique(cats_all, return_inverse=True)
    ncat = len(cat_list)
    ex = np.array([exemplar_number(x) for x in names])
    M = np.zeros((ncat, n), np.float32)
    M[cat_of, np.arange(n)] = 1.0
    M /= M.sum(1, keepdims=True)                      # category-mean operator
    iu = np.triu_indices(ncat, 1)
    print(f"[{sub}] {n} images, {ncat} categories, ROIs {rois}", flush=True)

    Xb = {r: z[r][keep].astype(np.float32) for r in rois}
    Rb = np.stack([zrank(dist_matrix(M @ Xb[r])[iu]) for r in rois])
    sb = {r: split_half_reliability(Xb[r], cats_all, ex, ncat) for r in rois}

    ml, rows_m = [], []
    for m in MODELS:
        d = f"{MULTI}/{m}"
        all_names = json.load(open(f"{d}/images.json"))
        pos = {x: i for i, x in enumerate(all_names)}
        sel = np.array([pos[x] for x in names])
        keys = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npy"))
        for key in keys:
            F = np.load(f"{d}/{key}.npy", mmap_mode="r")
            Fc = M @ np.asarray(F[sel], dtype=np.float32)      # pool per image (done), then average per category
            rows_m.append(zrank(dist_matrix(Fc)[iu]))
            ml.append((m, key))
        print(f"[{sub}] {m}: {len(keys)} readouts loaded", flush=True)
    Rm = np.stack(rows_m).astype(np.float64)
    ks = rng.multinomial(ncat, np.full(ncat, 1.0 / ncat), size=N_BOOT).astype(np.float64)
    est, boots = fast_boot(Rb.astype(np.float64), Rm, iu[0], iu[1], ks)
    lo, hi, _ = ci_p(boots)

    rows = []
    for i, r in enumerate(rois):
        for j, (m, key) in enumerate(ml):
            stage, block, grid = parse(key)
            rows.append(dict(roi=r, model=m, readout=key, stage=stage, block=block, grid=grid,
                             rho=est[i, j], ci_lo=lo[i, j], ci_hi=hi[i, j],
                             brain_reliability_SB=sb[r], ceiling=np.sqrt(sb[r]) if sb[r] > 0 else np.nan))
    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{RES}/{sub}_rsa_pooling.csv", index=False)

    idx = {x: j for j, x in enumerate(ml)}
    drows = []
    for a, b in PAIRS:
        if a not in MODELS or b not in MODELS:
            continue
        for (m, key), ja in idx.items():
            if m != a:
                continue
            jb = idx[(b, key)]
            dd = boots[:, :, ja] - boots[:, :, jb]
            dlo, dhi, dp = ci_p(dd)
            stage, block, grid = parse(key)
            for i, r in enumerate(rois):
                drows.append(dict(roi=r, contrast=f"{a}_minus_{b}", readout=key, stage=stage, block=block,
                                  grid=grid, diff=est[i, ja] - est[i, jb], ci_lo=dlo[i], ci_hi=dhi[i], p_boot=dp[i]))
    pd.DataFrame(drows).to_csv(f"{RES}/{sub}_rsa_pooling_diff.csv", index=False)
    print(f"[{sub}] saved {RES}/{sub}_rsa_pooling.csv and _diff.csv", flush=True)


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

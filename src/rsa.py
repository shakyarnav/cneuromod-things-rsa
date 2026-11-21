#!/usr/bin/env python
"""Model-brain RSA on CNeuroMod-THINGS ROI patterns.

Usage: python rsa.py sub-01 [sub-02 ...]
Env  : MODELS (default supervised,simclr,random), N_BOOT (1000), N_DRAW (10),
       THINGS_ROOT, RSA_OUT (brain npz), FEAT_OUT (feature npz), RES_OUT (results)
Reads : RSA_OUT/<sub>_roi_betas.npz, FEAT_OUT/<model>_features.npz, the annotations json
Writes: RES_OUT/<sub>_rsa_overall.csv, <sub>_rsa_clutter.csv, <sub>_rsa_model_diff.csv

Analyses (images = those with all 3 repetitions):
 1. Overall: category-level RDMs (mean pattern per category), Spearman model-vs-brain,
    bootstrap over categories.
 2. Clutter: category-MATCHED image-level RDMs. In each category that has both low- and high-clutter
    images, n_c = min(#low, #high) images are drawn from each side, so both conditions contain the same
    categories in the same numbers. Alignment is computed within each condition; the difference
    (high - low) gets a paired bootstrap over categories.
 3. Model differences (supervised - simclr) from the same bootstrap samples.
Spearman is computed as a weighted Pearson correlation of rank-transformed distances; the bootstrap
resamples categories by weighting each pair by the resampling counts of its two categories.
"""
import json
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.environ.get("THINGS_ROOT", "/scratch/arnav.shakya/cneuromod-things")
BRAIN = os.environ.get("RSA_OUT", "/scratch/arnav.shakya/rsa_work/brain")
FEAT = os.environ.get("FEAT_OUT", "/scratch/arnav.shakya/rsa_work/features")
RES = os.environ.get("RES_OUT", "/scratch/arnav.shakya/rsa_work/results")
ANN = f"{ROOT}/THINGS/glmsingle/task-things_imgAnnotations.json"
MODELS = os.environ.get("MODELS", "supervised,simclr,random").split(",")
LAYERS = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
N_BOOT = int(os.environ.get("N_BOOT", "1000"))
N_DRAW = int(os.environ.get("N_DRAW", "10"))
MIN_REPS = 3
MIN_CATS = 30
SEED = 0
# clutter level: 0 lone object, 1 minimal background, 2 rich background, 3 full scene
CONTRASTS = {"low_vs_high": ({0, 1}, {2, 3}), "lone_vs_scene": ({0}, {3})}


def flag(a, k):
    v = a.get(k)
    return int(v) if v is not None and v == v else 0


def clutter_level(a):
    if flag(a, "manual_scene"):
        return 3
    if flag(a, "manual_rich_background"):
        return 2
    if flag(a, "manual_lone_object"):
        return 0
    return 1


def exemplar_number(name):
    m = re.search(r"_(\d+)[a-z]?$", name)
    return int(m.group(1)) if m else -1


def dist_matrix(X):
    """Correlation distance between rows after removing each column's mean across rows."""
    X = np.asarray(X, np.float32)
    X = X - X.mean(0, keepdims=True)
    X = X - X.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return (1.0 - X @ X.T).astype(np.float32)


def zrank(v):
    r = rankdata(v)
    return (r - r.mean()) / r.std()


def wcorr(Rb, Rm, w):
    """Weighted Pearson between every row of Rb (n_roi x P) and every row of Rm (n_ml x P)."""
    W = w.sum()
    mb, mm = Rb @ w / W, Rm @ w / W
    cov = (Rb * w) @ Rm.T / W - np.outer(mb, mm)
    vb, vm = (Rb ** 2) @ w / W - mb ** 2, (Rm ** 2) @ w / W - mm ** 2
    return cov / np.sqrt(np.outer(vb, vm))


def boot_corr(Rb, Rm, ci0, ci1, ks):
    """Point estimate and category-bootstrap samples. ci0/ci1: category index of each pair's two items."""
    est = wcorr(Rb, Rm, np.ones(len(ci0)))
    boots = np.empty((len(ks),) + est.shape)
    for b, k in enumerate(ks):
        boots[b] = wcorr(Rb, Rm, k[ci0] * k[ci1])
    return est, boots


def ci_p(x):
    lo, hi = np.percentile(x, [2.5, 97.5], axis=0)
    p = 2 * np.minimum((x <= 0).mean(0), (x >= 0).mean(0))
    return lo, hi, np.maximum(p, 1.0 / (len(x) + 1))


def split_half_reliability(X, cats, ex, ncat):
    odd = ex % 2 == 1
    ok = sorted(set(cats[odd]) & set(cats[~odd]))
    if len(ok) < 20:
        return np.nan
    def rdm(sel):
        P = np.stack([X[(cats == c) & sel].mean(0) for c in ok])
        return dist_matrix(P)[np.triu_indices(len(ok), 1)]
    r = np.corrcoef(rankdata(rdm(odd)), rankdata(rdm(~odd)))[0, 1]
    return 2 * r / (1 + r)


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
    level = np.array([clutter_level(ann[x]) for x in names])
    print(f"[{sub}] {n} images (>= {MIN_REPS} reps), {ncat} categories | ROIs: {rois} | models: {MODELS}")
    print(f"[{sub}] clutter levels (0 lone, 1 minimal, 2 rich, 3 scene): {np.bincount(level, minlength=4).tolist()}")

    Xb = {r: z[r][keep].astype(np.float32) for r in rois}
    Fm = {}
    for m in MODELS:
        f = np.load(f"{FEAT}/{m}_features.npz")
        pos = {str(x): i for i, x in enumerate(f["images"])}
        sel = [pos[x] for x in names]
        for L in LAYERS:
            Fm[(m, L)] = f[L][sel]
    ml = list(Fm)
    os.makedirs(RES, exist_ok=True)

    # ---- 1. overall, category level ----
    cat_mean = lambda X: np.stack([X[cat_of == c].mean(0) for c in range(ncat)])
    iu = np.triu_indices(ncat, 1)
    Rb = np.stack([zrank(dist_matrix(cat_mean(Xb[r]))[iu]) for r in rois])
    Rm = np.stack([zrank(dist_matrix(cat_mean(Fm[k]))[iu]) for k in ml])
    ks = rng.multinomial(ncat, np.full(ncat, 1.0 / ncat), size=N_BOOT).astype(np.float64)
    est, boots = boot_corr(Rb, Rm, iu[0], iu[1], ks)
    lo, hi, _ = ci_p(boots)
    sb = {r: split_half_reliability(Xb[r], cats_all, ex, ncat) for r in rois}
    rows = [dict(roi=r, model=m, layer=L, rho=est[i, j], ci_lo=lo[i, j], ci_hi=hi[i, j],
                 brain_reliability_SB=sb[r], ceiling=np.sqrt(sb[r]) if sb[r] > 0 else np.nan)
            for i, r in enumerate(rois) for j, (m, L) in enumerate(ml)]
    overall = pd.DataFrame(rows)
    overall.to_csv(f"{RES}/{sub}_rsa_overall.csv", index=False)
    best = {}  # best layer per (roi, model) by overall alignment
    for (r, m), g in overall.groupby(["roi", "model"]):
        best[(r, m)] = g.loc[g.rho.idxmax(), "layer"]

    diff_rows = []
    def add_diffs(label, e, b):
        if "supervised" in MODELS and "simclr" in MODELS:
            for L in LAYERS:
                js, jc = ml.index(("supervised", L)), ml.index(("simclr", L))
                d = b[:, :, js] - b[:, :, jc]
                dlo, dhi, dp = ci_p(d)
                for i, r in enumerate(rois):
                    diff_rows.append(dict(analysis=label, roi=r, layer=L, supervised_minus_simclr=e[i, js] - e[i, jc],
                                          ci_lo=dlo[i], ci_hi=dhi[i], p_boot=dp[i]))
    add_diffs("overall", est, boots)

    print(f"\n[{sub}] OVERALL category-level alignment (Spearman rho at each ROI x model's best layer; 95% CI over categories)")
    print(f"{'ROI':6s}{'ceil':>6s}" + "".join(f"{m:>34s}" for m in MODELS))
    for r in rois:
        cells = []
        for m in MODELS:
            g = overall[(overall.roi == r) & (overall.model == m) & (overall.layer == best[(r, m)])].iloc[0]
            cells.append(f"{g.rho:6.3f} [{g.ci_lo:6.3f},{g.ci_hi:6.3f}] {best[(r, m)]:>8s}")
        print(f"{r:6s}{overall[overall.roi == r].ceiling.iloc[0]:6.2f}" + "".join(f"{c:>34s}" for c in cells))

    # ---- 2. clutter contrasts on category-matched image-level RDMs ----
    Db = {r: dist_matrix(Xb[r]) for r in rois}
    Dm = {k: dist_matrix(Fm[k]) for k in ml}
    clutter_rows = []
    B_d = max(1, N_BOOT // N_DRAW)
    for cname, (lo_lv, hi_lv) in CONTRASTS.items():
        is_lo, is_hi = np.isin(level, list(lo_lv)), np.isin(level, list(hi_lv))
        elig = [c for c in range(ncat) if (is_lo & (cat_of == c)).any() and (is_hi & (cat_of == c)).any()]
        if len(elig) < MIN_CATS:
            print(f"[{sub}] contrast {cname}: only {len(elig)} categories with both levels - skipped")
            continue
        estL, estH, bL, bH, n_items = [], [], [], [], []
        for d in range(N_DRAW):
            idxL, idxH, catm = [], [], []
            for ci, c in enumerate(elig):
                lo_i, hi_i = np.flatnonzero(is_lo & (cat_of == c)), np.flatnonzero(is_hi & (cat_of == c))
                k = min(len(lo_i), len(hi_i))
                idxL += list(rng.choice(lo_i, k, replace=False))
                idxH += list(rng.choice(hi_i, k, replace=False))
                catm += [ci] * k
            catm = np.array(catm)
            m_items = len(catm)
            n_items.append(m_items)
            iu2 = np.triu_indices(m_items, 1)
            ks2 = rng.multinomial(len(elig), np.full(len(elig), 1.0 / len(elig)), size=B_d).astype(np.float64)
            out = []
            for idx in (idxL, idxH):
                ix = np.ix_(idx, idx)
                Rb2 = np.stack([zrank(Db[r][ix][iu2]) for r in rois])
                Rm2 = np.stack([zrank(Dm[k][ix][iu2]) for k in ml])
                out.append(boot_corr(Rb2, Rm2, catm[iu2[0]], catm[iu2[1]], ks2))
            (eL, kL), (eH, kH) = out
            estL.append(eL), estH.append(eH), bL.append(kL), bH.append(kH)
        eL, eH = np.mean(estL, 0), np.mean(estH, 0)
        bL, bH = np.concatenate(bL), np.concatenate(bH)
        dlo, dhi, dp = ci_p(bH - bL)
        llo, lhi, _ = ci_p(bL)
        hlo, hhi, _ = ci_p(bH)
        for i, r in enumerate(rois):
            for j, (m, L) in enumerate(ml):
                clutter_rows.append(dict(contrast=cname, roi=r, model=m, layer=L, n_categories=len(elig),
                                         n_images_per_condition=int(np.mean(n_items)),
                                         rho_low=eL[i, j], rho_high=eH[i, j], delta_high_minus_low=eH[i, j] - eL[i, j],
                                         delta_ci_lo=dlo[i, j], delta_ci_hi=dhi[i, j], delta_p_boot=dp[i, j]))
        add_diffs(f"{cname}:low", eL, bL)
        add_diffs(f"{cname}:high", eH, bH)
        print(f"\n[{sub}] CLUTTER {cname}: {len(elig)} matched categories, ~{int(np.mean(n_items))} images per condition")
        print("rho in low-clutter vs high-clutter images, delta = high - low [95% CI], at each model's best overall layer")
        print(f"{'ROI':6s}{'model':>11s}{'layer':>9s}{'rho_low':>9s}{'rho_high':>9s}{'delta':>9s}{'CI':>18s}{'p':>8s}")
        for i, r in enumerate(rois):
            for m in MODELS:
                j = ml.index((m, best[(r, m)]))
                print(f"{r:6s}{m:>11s}{best[(r, m)]:>9s}{eL[i, j]:9.3f}{eH[i, j]:9.3f}{eH[i, j] - eL[i, j]:9.3f}"
                      f"  [{dlo[i, j]:6.3f},{dhi[i, j]:6.3f}]{dp[i, j]:8.3f}")
    pd.DataFrame(clutter_rows).to_csv(f"{RES}/{sub}_rsa_clutter.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(f"{RES}/{sub}_rsa_model_diff.csv", index=False)
    print(f"\n[{sub}] saved CSVs in {RES}  (p values are uncorrected; correct across ROI x layer when interpreting)")


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

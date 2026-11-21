import json, sys, numpy as np, os
root = "features/multi"
bad = False
for m in ["supervised", "simclr", "random"]:
    d = f"{root}/{m}"
    names = json.load(open(f"{d}/images.json"))
    n = len(names)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    print(f"== {m}: {len(files)} readouts, images.json has {n} images")
    for f in files:
        a = np.load(f"{d}/{f}", mmap_mode="r")
        if a.shape[0] != n:
            print("  SHAPE MISMATCH", f, a.shape); bad = True; continue
        # last 200 rows, plus 200 spread across the array
        idx = np.unique(np.r_[np.arange(n-200, n), np.linspace(0, n-1, 200).astype(int)])
        rows = np.asarray(a[idx]).reshape(len(idx), -1)
        z = int((np.abs(rows).sum(1) == 0).sum())
        nan = int(np.isnan(rows).any(1).sum())
        if z or nan:
            print(f"  {f}: {z} all-zero rows, {nan} NaN rows (of {len(idx)} sampled)"); bad = True
print("PROBLEMS FOUND" if bad else "ALL SAMPLED ROWS NONZERO AND FINITE")

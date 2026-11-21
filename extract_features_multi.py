#!/usr/bin/env python
"""Multi-depth, multi-resolution ResNet-50 features.
Usage: python extract_features_multi.py supervised [simclr random]   (env LIMIT=N for a smoke test)
Writes FEAT_OUT/multi/<model>/<readout>.npy  (n_images x D, float32) and images.json
Readouts: stem_g{1,4,8}; layer{1,2}_g{1,2,4,8}; layer{3,4}_g{1,2,4,7};
          layer{s}_b{i}_g{1,4} for every bottleneck block; avgpool
Whole images, 224x224 bicubic, no crop (identical preprocessing to extract_features.py).
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from extract_features import build_model, Images, ANN, IMG_ROOT, OUT

GRIDS = {"stem": (1, 4, 8), "layer1": (1, 2, 4, 8), "layer2": (1, 2, 4, 8),
         "layer3": (1, 2, 4, 7), "layer4": (1, 2, 4, 7)}
BLOCK_GRIDS = (1, 4)


def pool(x, g):
    return F.adaptive_avg_pool2d(x, g).flatten(1)


def forward_all(m, x):
    out = {}
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    for g in GRIDS["stem"]:
        out[f"stem_g{g}"] = pool(x, g)
    for name in ("layer1", "layer2", "layer3", "layer4"):
        for bi, blk in enumerate(getattr(m, name), 1):
            x = blk(x)
            for g in BLOCK_GRIDS:
                out[f"{name}_b{bi}_g{g}"] = pool(x, g)
        for g in GRIDS[name]:
            out[f"{name}_g{g}"] = pool(x, g)
    out["avgpool"] = x.mean((2, 3))
    return out


def main(models):
    ann = json.load(open(ANN))
    names = sorted(ann)
    limit = int(os.environ.get("LIMIT", "0"))
    if limit:
        names = names[:limit]
        print(f"SMOKE TEST: first {limit} images")
    paths = [f"{IMG_ROOT}/images_fmri/{ann[n]['image_category']}/{n}.jpg" for n in names]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:3]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    for name in models:
        print(f"== {name} | {len(names)} images | {device}", flush=True)
        d = f"{OUT}/multi/{name}"
        os.makedirs(d, exist_ok=True)
        model = build_model(name).eval().to(device)
        loader = DataLoader(Images(paths), batch_size=32, num_workers=2)  # no shuffle: order == names
        mm, i0, t0 = {}, 0, time.time()
        with torch.no_grad():
            for i, x in enumerate(loader):
                f = forward_all(model, x.to(device))
                if not mm:
                    for k, v in f.items():
                        mm[k] = np.lib.format.open_memmap(f"{d}/{k}.npy", mode="w+", dtype=np.float32,
                                                          shape=(len(paths), v.shape[1]))
                    print(f"  {len(mm)} readouts; dims e.g. layer4_g7={mm.get('layer4_g7', np.empty((0,0))).shape[1]}, "
                          f"layer1_g8={mm['layer1_g8'].shape[1]}", flush=True)
                n = x.shape[0]
                for k, v in f.items():
                    mm[k][i0:i0 + n] = v.cpu().numpy().astype(np.float32)
                i0 += n
                if (i + 1) % 20 == 0:
                    print(f"  {name}: {i0}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
        for v in mm.values():
            v.flush()
        json.dump(names, open(f"{d}/images.json", "w"))
        print(f"  {name}: done, {len(mm)} readouts in {d}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["supervised", "simclr", "random"])

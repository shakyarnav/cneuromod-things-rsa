#!/usr/bin/env python
"""Extract pooled ResNet-50 features for the 4,320 CNeuroMod-THINGS images.

Usage: python extract_features.py supervised simclr random
Env  : IMG_ROOT, ANN, FEAT_OUT, SIMCLR_CKPT (path to the SimCLR checkpoint), TORCH_HOME
Output: FEAT_OUT/<model>_features.npz with `images` and one array per layer
        (layer1-4: global-average-pooled to a 2x2 grid; avgpool: 2048-d global average)
Whole images are resized to 224x224 (no cropping), so scene clutter is preserved.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

IMG_ROOT = os.environ.get("IMG_ROOT", "/scratch/arnav.shakya/cneuromod-things/THINGS/fmriprep/sourcedata/things/stimuli")
ANN = os.environ.get("ANN", "/scratch/arnav.shakya/cneuromod-things/THINGS/glmsingle/task-things_imgAnnotations.json")
OUT = os.environ.get("FEAT_OUT", "/scratch/arnav.shakya/rsa_work/features")
SIMCLR_CKPT = os.environ.get("SIMCLR_CKPT", "")
LAYERS = ["layer1", "layer2", "layer3", "layer4", "avgpool"]

TFM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TFM(Image.open(self.paths[i]).convert("RGB"))


# ---------- weights ----------
def best_prefix(keys, target):
    """Find the key prefix (e.g. 'module.backbone') that makes most checkpoint keys match ResNet-50 keys."""
    votes = Counter()
    for k in keys:
        parts = k.split(".")
        for i in range(len(parts)):
            if ".".join(parts[i:]) in target:
                votes[".".join(parts[:i])] += 1
                break
    return votes.most_common(1)[0] if votes else (None, 0)


def remap_state_dict(sd, target_keys):
    prefix, n = best_prefix(list(sd.keys()), target_keys)
    core = {k for k in target_keys if not k.startswith("fc.") and not k.endswith("num_batches_tracked")}
    if prefix is None or n < 0.9 * len(core):
        raise RuntimeError(f"checkpoint keys do not look like a ResNet-50 (matched {n}/{len(core)}); "
                           f"sample keys: {list(sd.keys())[:8]}")
    out = {}
    for k, v in sd.items():
        if prefix and not k.startswith(prefix + "."):
            continue
        kk = k[len(prefix) + 1:] if prefix else k
        if kk in target_keys:
            out[kk] = v
    missing = sorted(core - set(out))
    if missing:
        raise RuntimeError(f"{len(missing)} ResNet-50 weights missing after remapping, e.g. {missing[:5]}")
    print(f"  SimCLR checkpoint: stripped prefix {prefix!r}, matched {len(out)} tensors "
          f"(ignored {len(sd) - len(out)} others such as projection head)")
    return out


def load_simclr_weights(model, path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"set SIMCLR_CKPT to the SimCLR checkpoint file (got {path!r})")
    # safe mode (no arbitrary pickled code); the checkpoint also stores its training args as an argparse.Namespace,
    # a plain settings container, so only that one class is allowlisted
    with torch.serialization.safe_globals([argparse.Namespace]):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck
    if isinstance(ck, dict):
        for key in ("state_dict", "model", "model_state_dict", "backbone", "encoder"):
            if key in ck and isinstance(ck[key], dict):
                sd = ck[key]
                break
    sd = remap_state_dict(sd, set(model.state_dict().keys()))
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("fc.") and not k.endswith("num_batches_tracked")]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"load_state_dict problems: missing {bad[:5]}, unexpected {res.unexpected_keys[:5]}")
    return model


def build_model(name):
    if name == "supervised":
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    if name == "simclr":
        return load_simclr_weights(resnet50(weights=None), SIMCLR_CKPT)
    if name == "random":
        torch.manual_seed(0)
        return resnet50(weights=None)
    raise ValueError(f"unknown model {name}")


# ---------- features ----------
def forward_features(m, x):
    f = {}
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    for name in ("layer1", "layer2", "layer3", "layer4"):
        x = getattr(m, name)(x)
        f[name] = F.adaptive_avg_pool2d(x, 2).flatten(1)
    f["avgpool"] = x.mean((2, 3))
    return f


def extract(name, paths, device, batch=32, workers=2):
    model = build_model(name).eval().to(device)
    feats = {k: [] for k in LAYERS}
    loader = DataLoader(Images(paths), batch_size=batch, num_workers=workers)
    t0 = time.time()
    with torch.no_grad():
        for i, x in enumerate(loader):
            f = forward_features(model, x.to(device))
            for k in LAYERS:
                feats[k].append(f[k].cpu().numpy().astype(np.float32))
            if (i + 1) % 20 == 0:
                print(f"  {name}: {min((i + 1) * batch, len(paths))}/{len(paths)} images "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return {k: np.concatenate(v) for k, v in feats.items()}


def category_index(X, cats):
    """Mean correlation distance between categories minus within category (sanity check)."""
    X = X - X.mean(0, keepdims=True)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    D = 1.0 - X @ X.T
    same = cats[:, None] == cats[None, :]
    n = len(cats)
    tot = D.sum(dtype=np.float64) - np.trace(D, dtype=np.float64)
    within = D[same].sum(dtype=np.float64) - np.trace(D, dtype=np.float64)
    n_within = int(same.sum()) - n
    return (tot - within) / (n * n - n - n_within) - within / n_within


def main(models):
    ann = json.load(open(ANN))
    names = sorted(ann)
    limit = int(os.environ.get("LIMIT", "0"))
    if limit:
        names = names[:limit]
        print(f"SMOKE TEST: first {limit} images only")
    paths = [f"{IMG_ROOT}/images_fmri/{ann[n]['image_category']}/{n}.jpg" for n in names]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:3]}")
    cats = np.array([ann[n]["image_category"] for n in names])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    print(f"{len(names)} images | device: {device} | threads: {torch.get_num_threads()}")
    os.makedirs(OUT, exist_ok=True)
    for name in models:
        print(f"== {name}")
        feats = extract(name, paths, device)
        np.savez_compressed(f"{OUT}/{name}_features.npz", images=np.array(names), **feats)
        idx = {k: round(float(category_index(v, cats)), 4) for k, v in feats.items()}
        print(f"  {name}: feature dims { {k: v.shape[1] for k, v in feats.items()} }")
        print(f"  {name}: category index per layer {idx}  (higher = more category structure)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["supervised", "simclr", "random"])

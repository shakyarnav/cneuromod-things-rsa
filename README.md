# DNN–brain RSA on CNeuroMod-THINGS
ResNet-50 (supervised ImageNet, SimCLR) vs fMRI ROI category RDMs in 4 subjects.

## Pipeline (run from repo root)
1. `src/extract_roi_betas.py` – ROI patterns from GLMsingle betas
2. `src/brain_rdm.py` – category RDMs, split-half reliability
3. `src/extract_features_multi.py` – ResNet-50 readouts (stem, layer1–4, blocks, avgpool; 1/2/4/8 grids)
4. `src/rsa_pool.py` – RSA + category bootstrap
5. `src/summarize_pool.py`, `src/sanity_gate.py` – summaries, check vs original 2x2 run
6. `src/pub_figs_v2.py`, `src/pub_figs_rdm.py` – Figs A, B, F

`archive/` holds earlier analyses (clutter contrasts, v1 figures) not used in the final results.
Data are not included (CNeuroMod-THINGS, research-only licence). Paths are hard-coded for Param Ananta.
Environments: `env/`.

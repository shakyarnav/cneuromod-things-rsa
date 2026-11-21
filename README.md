# DNN–brain RSA on CNeuroMod-THINGS
ResNet-50 (supervised ImageNet, SimCLR) vs fMRI ROI category RDMs in 4 subjects.

Pipeline order: extract_roi_betas.py -> brain_rdm.py -> extract_features_multi.py -> rsa_pool.py -> summarize_pool.py -> pub_figs_v2.py / pub_figs_rdm.py

Data are not included (CNeuroMod-THINGS, research-only licence). Paths are hard-coded for the Param Ananta cluster (/scratch/arnav.shakya/...).
Environments: see env_rsa_pip.txt and env_dnn_pip.txt.

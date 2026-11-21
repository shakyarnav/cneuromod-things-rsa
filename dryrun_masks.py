import glob, os, sys
import nibabel as nib
import extract_roi_betas as e

for sub in sys.argv[1:]:
    brp = f"{e.GS}/{sub}/glmsingle/input/{sub}_task-things_space-T1w_label-brain_desc-unionNonNaN_mask.nii"
    bshape = nib.load(brp).shape if os.path.exists(brp) else None
    print(f"== {sub} | beta-grid mask: {bshape if bshape else 'not downloaded yet'}")
    jobs = [(r, f"{e.FLOC}/{sub}/rois/task-derived/*_roi-{r}_*mask.nii.gz") for r in e.HIGH]
    if os.path.isdir(f"{e.RETINO}/{sub}"):
        jobs += [(r, f"{e.RETINO}/{sub}/rois/*res-func*label-{r}_desc-nn_mask.nii.gz") for r in e.EARLY]
    else:
        print("   (no retinotopy for this subject: V1-V3 will be skipped)")
    for roi, pat in jobs:
        try:
            p = e.pick(glob.glob(pat), f"{sub} {roi}")
            m = nib.load(p)
            ok = "" if bshape is None or m.shape == bshape else "  <-- GRID MISMATCH"
            print(f"   {roi:4s} {int((m.get_fdata() > 0.5).sum()):5d} vox  {os.path.basename(p)[:95]}{ok}")
        except Exception as ex:
            print(f"   {roi:4s} PROBLEM: {ex}")

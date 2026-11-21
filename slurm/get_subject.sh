#!/bin/bash
# usage: get_subject.sh sub-02
export PATH=/scratch/arnav.shakya/conda/envs/datalad/bin:$PATH
s=$1
cd /scratch/arnav.shakya/cneuromod-things/THINGS/glmsingle || exit 1
echo "$(date) start $s"
datalad get \
  $s/glmsingle/input/${s}_task-things_space-T1w_label-brain_desc-unionNonNaN_mask.nii \
  $s/glmsingle/output/${s}_task-things_space-T1w_model-fitHrfGLMdenoiseRR_stat-noiseCeilings_statmap.nii.gz \
  $s/glmsingle/output/${s}_task-things_space-T1w_model-fitHrfGLMdenoiseRR_stat-imageBetas_desc-zscore_statseries.h5
echo "$(date) finished $s (datalad exit code $?)"

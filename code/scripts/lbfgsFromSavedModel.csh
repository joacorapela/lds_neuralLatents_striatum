#!/bin/csh

ipython --pdb doEstimateLBFGSfromSavedModel.py -- \
    --start_time_sec=5512.0 \
    --duration_sec=120 \
    --bin_size=0.02 \
    --max_iter=5000 \
    --lr=0.1 \
    --tolerance_grad=1e-2 \
    --tolerance_change=1e-4 \
    --skip_estimation_m0 \
    --skip_estimation_u \
    --clustersIndices_filename=../../metadata/clustersIndices_124_223.ini \
    --saved_model_filename=../../results/82227365_estimation.pickle

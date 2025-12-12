#!/bin/csh

ipython --pdb doEstimateEM.py -- \
    --start_time_sec=2240.0 \
    --duration_sec=120 \
    --bin_size=0.02 \
    --max_iter=5000 \
    --tol=0.01 \
    --skip_estimation_m0 \
    --skip_estimation_u \
    --clustersIndices_filename=../../metadata/clustersIndices_124_223.ini \
    --initialConditions_params_filename=../../metadata/00000013_initialConditions.ini

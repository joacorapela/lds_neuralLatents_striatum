#!/bin/csh

ipython --pdb doWriteInitialConditionsFile.py -- \
    --n_latents=5 \
    --n_clusters=100 \
    --initial_conditions_params_file_num=13 \
    --initialConditions_params_filename_pattern="../../metadata/{:08d}_initialConditions.ini" \
    --constant_u=0.0 \
    --mean_B=0.03 \
    --sigma_B=0.01 \
    --mean_Q=0.09 \
    --sigma_Q=0.01 \
    --mean_a=0.1 \
    --sigma_a=0.01 \
    --mean_Z=0.07 \
    --sigma_Z=0.02 \
    --mean_R=0.10 \
    --sigma_R=0.02 \
    --constant_m0=0.0 \
    --mean_V0=0.1 \
    --sigma_V0=0.01

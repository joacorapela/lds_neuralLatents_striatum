
import sys
import os
import random
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd

import ssm.learning
import ssm.utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time to plot (sec)",
                        type=float, default=2240.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        type=float, default=120.0)
    parser.add_argument("--bin_size", help="bin size (secs)", type=float,
                        default=0.02)
    parser.add_argument("--max_iter", help="maximum number of iterations", type=int,
                        default=5000)
    parser.add_argument("--tol", help="estimation convergence tolerance", type=float,
                        default=1e-1)
    parser.add_argument("--skip_sqrt_transform",
                        help="sqrt transform spike counts", action="store_true")
    parser.add_argument("--skip_estimation_B",
                        help="use this option to skip the estimation of B",
                        action="store_true")
    parser.add_argument("--skip_estimation_Q",
                        help="use this option to skip the estimation of Q",
                        action="store_true")
    parser.add_argument("--skip_estimation_Z",
                        help="use this option to skip the estimation of Z",
                        action="store_true")
    parser.add_argument("--skip_estimation_diag_R",
                        help="use this option to skip the estimation of diag_R",
                        action="store_true")
    parser.add_argument("--estimate_R",
                        help="use this option to estimate R",
                        action="store_true")
    parser.add_argument("--skip_estimation_m0",
                        help="use this option to skip the estimation of m0",
                        action="store_true")
    parser.add_argument("--skip_estimation_V0",
                        help="use this option to skip the estimation of V0",
                        action="store_true")
    parser.add_argument("--clustersIndices_filename",
                        help="filename with clusterIndices used for estimation", type=str,
                        default="../../metadata/clustersIndices_124_223.ini")
    parser.add_argument("--binned_spikes_filename_pattern",
                        help="binned spikes filename pattern", type=str,
                        default="../../results/binned_spikes_binSize{:.2f}_skipSqrtTrans{:d}.npz")
    parser.add_argument("--initialConditions_params_filename", type=str,
                        help="initial conditions filename",
                        default="../../metadata/00000009_initialConditions.ini")
    parser.add_argument("--results_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.{:s}")
    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    bin_size = args.bin_size
    max_iter = args.max_iter
    tol = args.tol
    skip_sqrt_transform = args.skip_sqrt_transform
    skip_estimation_B = args.skip_estimation_B
    skip_estimation_Q = args.skip_estimation_Q
    skip_estimation_Z = args.skip_estimation_Z
    skip_estimation_diag_R = args.skip_estimation_diag_R
    estimate_R = args.estimate_R
    skip_estimation_m0 = args.skip_estimation_m0
    skip_estimation_V0 = args.skip_estimation_V0
    clustersIndices_filename = args.clustersIndices_filename
    binned_spikes_filename_pattern = args.binned_spikes_filename_pattern
    initialConditions_params_filename = args.initialConditions_params_filename
    results_filename_pattern = args.results_filename_pattern

    clustersIndices = pd.read_csv(clustersIndices_filename, header=None, index_col=False).squeeze()

    binned_spikes_filename = binned_spikes_filename_pattern.format(
        bin_size, skip_sqrt_transform)
    load_res = np.load(binned_spikes_filename)

    data = load_res["binned_spikes"].T
    bins_centers = load_res["bins_centers"]

    valid_bins_mask = np.logical_and(start_time_sec<=bins_centers,
                                     bins_centers<end_time_sec)
    bins_centers = bins_centers[valid_bins_mask]
    data = data[valid_bins_mask, :]

    # make sure that the first data point is not NaN
    first_not_nan_index = np.where(~np.isnan(data).any(axis=1))[0][0]
    data = data[first_not_nan_index:,clustersIndices]
    #

    initialConditions_params = configparser.ConfigParser()
    initialConditions_params.read(initialConditions_params_filename)

    if not skip_estimation_B:
        aux = initialConditions_params['params']['B']
        B0 = ssm.utils.string_to_matrix(aux)

    if not skip_estimation_Z:
        aux = initialConditions_params['params']['Z']
        Z0 = ssm.utils.string_to_matrix(aux)

    if not skip_estimation_Q:
        aux = initialConditions_params['params']['Q']
        Q0 = ssm.utils.string_to_matrix(aux)

    if not skip_estimation_diag_R:
        aux = initialConditions_params['params']['diag_R']
        R0 = np.diag(ssm.utils.string_to_array1d(aux).squeeze())

    if estimate_R:
        aux = initialConditions_params['params']['R']
        R0 = ssm.utils.string_to_matrix(aux)

    if not skip_estimation_m0:
        aux = initialConditions_params['params']['m0']
        m0_0 = ssm.utils.string_to_array1d(aux).squeeze()

    if not skip_estimation_V0:
        aux = initialConditions_params['params']['V0']
        V0_0 = ssm.utils.string_to_matrix(aux)

    vars_to_estimate = {}

    if skip_estimation_B:
        vars_to_estimate["B"] = False
    else:
        vars_to_estimate["B"] = True

    if skip_estimation_Q:
        vars_to_estimate["Q"] = False
    else:
        vars_to_estimate["Q"] = True

    if skip_estimation_Z:
        vars_to_estimate["Z"] = False
    else:
        vars_to_estimate["Z"] = True

    if estimate_R:
        vars_to_estimate["R"] = True
    else:
        vars_to_estimate["R"] = False

    if skip_estimation_diag_R:
        vars_to_estimate["diag_R"] = False
    else:
        vars_to_estimate["diag_R"] = True

    if skip_estimation_m0:
        vars_to_estimate["m0"] = False
    else:
        vars_to_estimate["m0"] = True

    if skip_estimation_V0:
        vars_to_estimate["V0"] = False
    else:
        vars_to_estimate["V0"] = True

    if len(vars_to_estimate) == 0:
        ValueError("No variable to estimate.")

    optim_res = ssm.learning.em_SS_LDS(
        y=data.T, B0=B0, Q0=Q0, Z0=Z0, R0=R0,
        m0_0=m0_0, V0_0=V0_0, max_iter=max_iter, tol=tol,
        vars_to_estimate=vars_to_estimate,
    )

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_number, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False
    results_filename = results_filename_pattern.format(res_number, "pickle")

    with open(results_filename, "wb") as f:
        pickle.dump(optim_res, f)
    print(f"Saved Kalman filter results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "start_time_sec": start_time_sec,
        "duration_sec": args.duration_sec,
        "bin_size": bin_size,
        "max_iter": max_iter,
        "tol": tol,
        "skip_sqrt_transform": skip_sqrt_transform,
        "skip_estimation_B": skip_estimation_B,
        "skip_estimation_Q": skip_estimation_Q,
        "skip_estimation_Z": skip_estimation_Z,
        "skip_estimation_diag_R": skip_estimation_diag_R,
        "estimate_R": estimate_R,
        "skip_estimation_m0": skip_estimation_m0,
        "skip_estimation_V0": skip_estimation_V0,
        "clustersIndices_filename": clustersIndices_filename,
        "binned_spikes_filename": binned_spikes_filename,
        "initialConditions_params_filename": initialConditions_params_filename,
        "results_filename_pattern": results_filename_pattern,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

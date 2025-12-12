
import sys
import os
import random
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd
import torch

import ssm.learning
import ssm.utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time to plot (sec)",
                        type=float, default=5512.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        type=float, default=120.0)
    parser.add_argument("--bin_size", help="bin size (secs)", type=float,
                        default=0.02)
    parser.add_argument("--max_iter", help="maximum number of iterations", type=int,
                        default=5000)
    parser.add_argument("--lr", help="learning rate", type=float, default=1.0)
    parser.add_argument("--tolerance_grad",
                        help="convergence tolerance value for grandients",
                        type=float, default=1e-2)
    parser.add_argument("--tolerance_change",
                        help="convergence tolerance value for optimisation criterion values",
                        type=float, default=1e-4)
    parser.add_argument("--skip_sqrt_transform",
                        help="sqrt transform spike counts", action="store_true")
    parser.add_argument("--skip_estimation_u",
                        help="use this option to skip the estimation of u",
                        action="store_true")
    parser.add_argument("--skip_estimation_B",
                        help="use this option to skip the estimation of B",
                        action="store_true")
    parser.add_argument("--skip_estimation_Q",
                        help="use this option to skip the estimation of Q",
                        action="store_true")
    parser.add_argument("--skip_estimation_a",
                        help="use this option to skip the estimation of a",
                        action="store_true")
    parser.add_argument("--skip_estimation_Z",
                        help="use this option to skip the estimation of Z",
                        action="store_true")
    parser.add_argument("--skip_estimation_R",
                        help="use this option to skip the estimation of R",
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
    lr = args.lr
    tolerance_grad = args.tolerance_grad
    tolerance_change = args.tolerance_change
    skip_sqrt_transform = args.skip_sqrt_transform
    skip_estimation_u = args.skip_estimation_u
    skip_estimation_B = args.skip_estimation_B
    skip_estimation_Q = args.skip_estimation_Q
    skip_estimation_a = args.skip_estimation_a
    skip_estimation_Z = args.skip_estimation_Z
    skip_estimation_R = args.skip_estimation_R
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

    aux = raw_data = initialConditions_params['params']['u']
    u0 = ssm.utils.string_to_array1d(aux).squeeze()

    aux = initialConditions_params['params']['B']
    B0 = ssm.utils.string_to_matrix(aux)

    aux = raw_data = initialConditions_params['params']['a']
    a0 = ssm.utils.string_to_array1d(aux).squeeze()

    aux = initialConditions_params['params']['Z']
    Z0 = ssm.utils.string_to_matrix(aux)

    aux = initialConditions_params['params']['Q']
    Q0 = ssm.utils.string_to_matrix(aux)
    sqrt_diag_Q0 = np.sqrt(np.diag(Q0))

    aux = initialConditions_params['params']['diag_R']
    diag_R = ssm.utils.string_to_array1d(aux).squeeze()
    sqrt_diag_R0 = np.sqrt(diag_R)

    aux = initialConditions_params['params']['m0']
    m0_0 = ssm.utils.string_to_array1d(aux).squeeze()

    aux = initialConditions_params['params']['V0']
    V0_0 = ssm.utils.string_to_matrix(aux)
    sqrt_diag_V0_0 = np.sqrt(np.diag(V0_0))

    vars_to_estimate = {}

    if skip_estimation_u:
        vars_to_estimate["u"] = False
    else:
        vars_to_estimate["u"] = True

    if skip_estimation_B:
        vars_to_estimate["B"] = False
    else:
        vars_to_estimate["B"] = True

    if skip_estimation_Q:
        vars_to_estimate["sqrt_diag_Q"] = False
    else:
        vars_to_estimate["sqrt_diag_Q"] = True

    if skip_estimation_a:
        vars_to_estimate["a"] = False
    else:
        vars_to_estimate["a"] = True

    if skip_estimation_Z:
        vars_to_estimate["Z"] = False
    else:
        vars_to_estimate["Z"] = True

    if skip_estimation_R:
        vars_to_estimate["sqrt_diag_R"] = False
    else:
        vars_to_estimate["sqrt_diag_R"] = True

    if skip_estimation_m0:
        vars_to_estimate["m0"] = False
    else:
        vars_to_estimate["m0"] = True

    if skip_estimation_V0:
        vars_to_estimate["sqrt_diag_V0"] = False
    else:
        vars_to_estimate["sqrt_diag_V0"] = True

    if len(vars_to_estimate) == 0:
        ValueError("No variable to estimate.")

    data = torch.from_numpy(data)
    u0 = torch.from_numpy(u0)
    B0 = torch.from_numpy(B0)
    sqrt_diag_Q0 = torch.from_numpy(sqrt_diag_Q0)
    Q0 = torch.diag(sqrt_diag_Q0**2)
    a0 = torch.from_numpy(a0)
    Z0 = torch.from_numpy(Z0)
    sqrt_diag_R0 = torch.from_numpy(sqrt_diag_R0)
    m0_0 = torch.from_numpy(m0_0)
    sqrt_diag_V0_0 = torch.from_numpy(sqrt_diag_V0_0)

    optim_res = ssm.learning.torch_lbfgs_optimize_SS_LDS(
        y=data.T, u0=u0, B0=B0, sqrt_diag_Q0=sqrt_diag_Q0, a0=a0, Z0=Z0,
        sqrt_diag_R0=sqrt_diag_R0, m0_0=m0_0, sqrt_diag_V0_0=sqrt_diag_V0_0,
        max_iter=max_iter, lr=0.1, tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change, vars_to_estimate=vars_to_estimate,
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
        "lr": lr,
        "tolerance_grad": tolerance_grad,
        "tolerance_change": tolerance_change,
        "skip_sqrt_transform": skip_sqrt_transform,
        "skip_estimation_u": skip_estimation_u,
        "skip_estimation_B": skip_estimation_B,
        "skip_estimation_Q": skip_estimation_Q,
        "skip_estimation_a": skip_estimation_a,
        "skip_estimation_Z": skip_estimation_Z,
        "skip_estimation_R": skip_estimation_R,
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

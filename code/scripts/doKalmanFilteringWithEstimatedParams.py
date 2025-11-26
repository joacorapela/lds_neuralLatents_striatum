
import sys
import os
import random
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd

import ssm.inference


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time to plot (sec)",
                        type=float, default=2240.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        type=float, default=120.0)
    parser.add_argument("--est_res_num", type=int,
                        help="estimation result number",
                        default=58110546)
    parser.add_argument("--clustersIndices_filename",
                        help="filename with clusterIndices used for estimation", type=str,
                        default="../../metadata/clustersIndices_124_223.ini")
    parser.add_argument("--est_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.{:s}")
    parser.add_argument("--results_filename_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}")
    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    est_res_num = args.est_res_num
    clustersIndices_filename = args.clustersIndices_filename
    est_filename_pattern = args.est_filename_pattern
    results_filename_pattern = args.results_filename_pattern

    est_metadata_filename = est_filename_pattern.format(est_res_num, "ini")
    est_metadata = configparser.ConfigParser()
    est_metadata.read(est_metadata_filename)
    binned_spikes_filename = est_metadata["params"]["binned_spikes_filename"]

    clustersIndices = pd.read_csv(clustersIndices_filename, header=None, index_col=False).squeeze()

    est_res_filename = est_filename_pattern.format(est_res_num, "pickle")
    with open(est_res_filename, "rb") as f:
        optim_res = pickle.load(f)

    B = optim_res["B"]
    Q = optim_res["Q"]
    Z = optim_res["Z"]
    R = np.diag(optim_res["diag_R"])
    m0 = optim_res["m0"]
    V0 = optim_res["V0"]

    load_res = np.load(binned_spikes_filename)
    data = load_res["binned_spikes"].T
    bins_centers = load_res["bins_centers"]

    # make sure that the first data point is not NaN
    first_not_nan_index = np.where(~np.isnan(data).any(axis=1))[0][0]
    data = data[first_not_nan_index:,clustersIndices]
    #

    valid_bins_mask = np.logical_and(start_time_sec<=bins_centers,
                                     bins_centers<end_time_sec)
    bins_centers = bins_centers[valid_bins_mask]
    data = data[valid_bins_mask, :]

    filter_res = ssm.inference.filterLDS_SS_withMissingValues_np(
        y=data.T, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_num = random.randint(0, 10**8)
        metadata_filename = results_filename_pattern.format(res_num, "ini")
        if not os.path.exists(metadata_filename):
            res_prefix_used = False
    results_filename = results_filename_pattern.format(res_num, "pickle")

    results = dict(xnn1=filter_res["xnn1"], Pnn1=filter_res["Pnn1"],
                   xnn=filter_res["xnn"], Pnn=filter_res["Pnn"], 
                   bins_centers=bins_centers, logLike=filter_res["logLike"],
                   Z=Z)
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved Kalman filter results to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "est_res_num": est_res_num,
        "est_filename_pattern": est_filename_pattern,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)


import sys
import os
import random
import argparse
import configparser
import numpy as np

import ssm.utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_latents", type=int, help="number of latents",
                        default=10)
    parser.add_argument("--n_clusters", type=int, help="number of clusters",
                        default=84)
    parser.add_argument("--initial_conditions_params_file_num", type=int,
                        help="initial conditions file number", default=0)
    parser.add_argument("--initialConditions_params_filename_pattern", type=str,
                        default="../../metadata/{:08d}_initialConditions.ini",
                        help="initial conditions filename pattern")
    parser.add_argument("--sigma_B", type=float, default=0.1,
                        help="standard deviation of the normal distribution for the diagonal of the transition matrix B")
    parser.add_argument("--sigma_Z", type=float, default=0.1,
                        help="standard deviation of the normal distribution for the elements of the latents to observation transformation matrix Z")
    parser.add_argument("--mean_Q", type=float, default=0.1,
                        help="mean of the normal distribution for the diagonal of the state noise covariance Q")
    parser.add_argument("--sigma_Q", type=float, default=0.02,
                        help="standard deviation of the normal distribution for the diagonal of the state noise covariance Q")
    parser.add_argument("--mean_R", type=float, default=0.1,
                        help="mean of the normal distribution for the diagonal of the state noise covariance R")
    parser.add_argument("--sigma_R", type=float, default=0.02,
                        help="standard deviation of the normal distribution for the diagonal of the measurement noise covariance R")
    parser.add_argument("--sigma_m0", type=float, default=0.1,
                        help="standar deviation of the normal distribution for the elements of the mean of the initial state m0")
    parser.add_argument("--mean_V0", type=float, default=0.1,
                        help="mean of the normal distribution for the diagonal of the initial state covariance V0")
    parser.add_argument("--sigma_V0", type=float, default=0.02,
                        help="standard deviation of the normal distribution for the diagonal of the initial state covariance V0")
    args = parser.parse_args()

    n_latents = args.n_latents
    n_clusters = args.n_clusters
    initial_conditions_params_file_num = args.initial_conditions_params_file_num
    initialConditions_params_filename = args.initialConditions_params_filename_pattern.format(initial_conditions_params_file_num)
    sigma_B = args.sigma_B
    sigma_Z = args.sigma_Z
    mean_Q = args.mean_Q
    sigma_Q = args.sigma_Q
    mean_R = args.mean_R
    sigma_R = args.sigma_R
    sigma_m0 = args.sigma_m0
    mean_V0 = args.mean_V0
    sigma_V0 = args.sigma_V0

    B = np.diag(np.random.normal(loc=0, scale=sigma_B, size=n_latents))
    Z = np.random.normal(loc=0, scale=sigma_Z, size=(n_clusters, n_latents))
    Q = np.diag(np.abs(np.random.normal(loc=mean_Q, scale=sigma_Q, size=n_latents)))
    diag_R = np.abs(np.random.normal(loc=mean_R, scale=sigma_R, size=n_clusters))
    m0 = np.random.normal(loc=0, scale=sigma_m0, size=n_latents)
    V0 = np.diag(np.abs(np.random.normal(loc=mean_V0, scale=sigma_V0, size=n_latents)))

    initialConditions_params = configparser.ConfigParser()
    initialConditions_params["params"] = {"m0": ssm.utils.array1d_to_string(m0),
                                          "V0": ssm.utils.matrix_to_string(V0),
                                          "B": ssm.utils.matrix_to_string(B),
                                          "Q": ssm.utils.matrix_to_string(Q),
                                          "Z": ssm.utils.matrix_to_string(Z),
                                          "diag_R": ssm.utils.array1d_to_string(diag_R),
                                          }
    if not os.path.exists(initialConditions_params_filename):
        with open(initialConditions_params_filename, "w") as f:
            initialConditions_params.write(f)
    else:
        raise RuntimeError(f"Initial conditions parameters file {initialConditions_params_filename} exists")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

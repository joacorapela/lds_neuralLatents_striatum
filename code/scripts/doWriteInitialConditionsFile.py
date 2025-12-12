
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
                        default=5)
    parser.add_argument("--n_clusters", type=int, help="number of clusters",
                        default=100)
    parser.add_argument("--initial_conditions_params_file_num", type=int,
                        help="initial conditions file number", default=12)
    parser.add_argument("--initialConditions_params_filename_pattern", type=str,
                        default="../../metadata/{:08d}_initialConditions.ini",
                        help="initial conditions filename pattern")
    parser.add_argument("--constant_u", type=float, default=0.0,
                        help="constant value for the state offset")
    parser.add_argument("--mean_B", type=float, default=0.1,
                        help="mean of the normal distribution of the transition matrix B")
    parser.add_argument("--sigma_B", type=float, default=0.3,
                        help="standard deviation of the normal distribution of the transition matrix B")
    parser.add_argument("--mean_Q", type=float, default=0.3,
                        help="mean of the normal distribution for the diagonal of the state noise covariance Q")
    parser.add_argument("--sigma_Q", type=float, default=0.1,
                        help="standard deviation of the normal distribution for the diagonal of the state noise covariance Q")
    parser.add_argument("--mean_a", type=float, default=1.0,
                        help="mean of the observations offset")
    parser.add_argument("--sigma_a", type=float, default=0.1,
                        help="standard deviation of the observations offset")
    parser.add_argument("--mean_Z", type=float, default=0.05,
                        help="mean of the normal distribution for the elements of the latents to observation transformation matrix Z")
    parser.add_argument("--sigma_Z", type=float, default=0.01,
                        help="standard deviation of the normal distribution for the elements of the latents to observation transformation matrix Z")
    parser.add_argument("--mean_R", type=float, default=0.5,
                        help="mean of the normal distribution for the diagonal of the state noise covariance R")
    parser.add_argument("--sigma_R", type=float, default=0.1,
                        help="standard deviation of the normal distribution for the diagonal of the measurement noise covariance R")
    parser.add_argument("--constant_m0", type=float, default=0.0,
                        help="constant value for the state initial mean")
    parser.add_argument("--mean_V0", type=float, default=0.02,
                        help="mean of the normal distribution for the diagonal of the initial state covariance V0")
    parser.add_argument("--sigma_V0", type=float, default=0.005,
                        help="standard deviation of the normal distribution for the diagonal of the initial state covariance V0")
    args = parser.parse_args()

    n_latents = args.n_latents
    n_clusters = args.n_clusters
    initial_conditions_params_file_num = args.initial_conditions_params_file_num
    initialConditions_params_filename = args.initialConditions_params_filename_pattern.format(initial_conditions_params_file_num)

    if os.path.exists(initialConditions_params_filename):
        raise RuntimeError(f"Initial conditions parameters file {initialConditions_params_filename} exists")

    constant_u = args.constant_u
    mean_B = args.mean_B
    sigma_B = args.sigma_B
    mean_a = args.mean_a
    sigma_a = args.sigma_a
    mean_Z = args.mean_Z
    sigma_Z = args.sigma_Z
    mean_Q = args.mean_Q
    sigma_Q = args.sigma_Q
    mean_R = args.mean_R
    sigma_R = args.sigma_R
    constant_m0 = args.constant_m0
    mean_V0 = args.mean_V0
    sigma_V0 = args.sigma_V0

    u = constant_u * np.ones(shape=n_latents)
    B = np.random.normal(loc=mean_B, scale=sigma_B, size=(n_latents, n_latents))
    a = np.random.normal(loc=mean_a, scale=sigma_a, size=n_clusters)
    Z = np.random.normal(loc=mean_Z, scale=sigma_Z, size=(n_clusters, n_latents))
    Q = np.diag(np.abs(np.random.normal(loc=mean_Q, scale=sigma_Q, size=n_latents)))
    diag_R = np.abs(np.random.normal(loc=mean_R, scale=sigma_R, size=n_clusters))
    m0 = constant_m0 * np.ones(shape=n_latents)
    V0 = np.diag(np.abs(np.random.normal(loc=mean_V0, scale=sigma_V0, size=n_latents)))

    initialConditions_params = configparser.ConfigParser()
    initialConditions_params["params"] = {"m0": ssm.utils.array1d_to_string(m0),
                                          "V0": ssm.utils.matrix_to_string(V0),
                                          "u": ssm.utils.array1d_to_string(u),
                                          "B": ssm.utils.matrix_to_string(B),
                                          "Q": ssm.utils.matrix_to_string(Q),
                                          "a": ssm.utils.array1d_to_string(a),
                                          "Z": ssm.utils.matrix_to_string(Z),
                                          "diag_R": ssm.utils.array1d_to_string(diag_R),
                                          }
    with open(initialConditions_params_filename, "w") as f:
        initialConditions_params.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

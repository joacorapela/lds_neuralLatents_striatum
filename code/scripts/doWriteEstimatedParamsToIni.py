
import sys
import pickle
import argparse
import configparser
import numpy as np


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_num", type=int,
                        help="estimation result number",
                        default=80065003)
    parser.add_argument("--est_res_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.pickle")
    parser.add_argument("--est_params_filename_pattern", type=str,
                        default="../../results/{:08d}_estimated_params.ini")
    args = parser.parse_args()

    est_res_num = args.est_res_num
    est_res_filename_pattern = args.est_res_filename_pattern
    est_params_filename_pattern = args.est_params_filename_pattern

    est_res_filename = est_res_filename_pattern.format(est_res_num)
    est_params_filename = est_params_filename_pattern.format(est_res_num)

    with open(est_res_filename, "rb") as f:
        est_results = pickle.load(f)

    ini = configparser.ConfigParser()
    ini["params"] = {
        "m0": est_results["m0"],
        "V0": est_results["V0"],
        "u": est_results["u"],
        "B": est_results["B"],
        "Q": est_results["Q"],
        "a": est_results["a"],
        "Z": est_results["Z"],
        "diag_R": np.diag(est_results["R"]),
    }
    with open(est_params_filename, "w") as f:
        ini.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

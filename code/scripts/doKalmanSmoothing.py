
import sys
import pickle
import argparse
import configparser

import ssm.inference


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtering_res_num", type=int,
                        help="filtering result number",
                        default=78991310)
    parser.add_argument("--results_filename_pattern", type=str,
                        default="../../results/{:08d}_{:s}.{:s}")
    args = parser.parse_args()

    filtering_res_num = args.filtering_res_num
    results_filename_pattern = args.results_filename_pattern

    filtering_filename = results_filename_pattern.format(filtering_res_num,
                                                         "filtered", "pickle")
    smoothing_filename = results_filename_pattern.format(filtering_res_num,
                                                         "smoothed", "pickle")

    filtered_metadata_filename = \
        results_filename_pattern.format(filtering_res_num, "filtered", "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)
    est_res_num = int(filtered_metadata["params"]["est_res_num"])
    est_filename_pattern = filtered_metadata["params"]["est_filename_pattern"]
    est_filename = est_filename_pattern.format(est_res_num, "pickle")
    with open(est_filename, "rb") as f:
        optim_res = pickle.load(f)

    with open(filtering_filename, "rb") as f:
        filtering_res = pickle.load(f)

    smoothing_res = ssm.inference.smoothLDS_SS(
        B=optim_res["B"], xnn=filtering_res["xnn"], Pnn=filtering_res["Pnn"],
        xnn1=filtering_res["xnn1"], Pnn1=filtering_res["Pnn1"],
        m0=optim_res["m0"], V0=optim_res["V0"])

    with open(smoothing_filename, "wb") as f:
        pickle.dump(smoothing_res, f)

    print(f"Saved Kalman smoothing results to {smoothing_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

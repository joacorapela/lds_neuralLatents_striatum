
import sys
import pickle
import argparse
import configparser
import numpy as np

import ssm.neural_latents.utils
import hmm.inference
import hmmUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_start_time_sec", help="train start time to plot (sec)",
                        type=float, default=4045.0)
    parser.add_argument("--train_duration_sec", help="train_duration to plot (sec)",
                        type=float, default=300.0)
    parser.add_argument("--train_filtering_res_number", type=int,
                        help="number corresponding to filtered results filename used to learn the params hmm",
                        default=42904130)
    parser.add_argument("--test_filtering_res_number", type=int,
                        help="number corresponding to filtered results filename used to test the hmm model",
                        # default=42904130)
                        # default=12157029)
                        # default=21792914)
                        default=37772953)
    parser.add_argument("--filtering_res_filenames_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_filtered.{:s}")
    parser.add_argument("--hmm_params_filename_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_params_from{:.02f}_to{:.02f}.npz")
    parser.add_argument("--most_prob_states_seq_filename_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/train{:08d}_test{:08d}_hmm_most_prob_state_seq_trainFrom{:.02f}_trainTo{:.02f}_testFrom{:.02f}_testTo{:.02f}.npz")

    args = parser.parse_args()

    train_start_time_sec = args.train_start_time_sec
    train_end_time_sec = args.train_start_time_sec + args.train_duration_sec
    train_filtering_res_number = args.train_filtering_res_number
    test_filtering_res_number = args.test_filtering_res_number
    filtering_res_filenames_pattern = \
        args.filtering_res_filenames_pattern
    hmm_params_filename = args.hmm_params_filename_pattern.format(
        train_filtering_res_number, train_start_time_sec, train_end_time_sec)

    test_filtered_metadata_filename = \
        filtering_res_filenames_pattern.format(test_filtering_res_number, "ini")
    test_filtered_metadata = configparser.ConfigParser()
    test_filtered_metadata.read(test_filtered_metadata_filename)

    test_start_time_sec = float(test_filtered_metadata["params"]["start_time_sec"])
    test_end_time_sec = float(test_filtered_metadata["params"]["end_time_sec"])

    most_prob_states_seq_filename = args.most_prob_states_seq_filename_pattern.format(
        train_filtering_res_number, test_filtering_res_number, train_start_time_sec, train_end_time_sec, test_start_time_sec, test_end_time_sec)

    test_filtering_res_filename = \
        filtering_res_filenames_pattern.format(test_filtering_res_number, "pickle")
    with open(test_filtering_res_filename, "rb") as f:
        test_filtering_res = pickle.load(f)

    train_filtered_metadata_filename = \
        filtering_res_filenames_pattern.format(train_filtering_res_number, "ini")
    train_filtered_metadata = configparser.ConfigParser()
    train_filtered_metadata.read(train_filtered_metadata_filename)
    train_est_filename = \
        train_filtered_metadata["params"]["est_filename_pattern"].format(int(train_filtered_metadata["params"]["est_res_num"]), "pickle")
    with open(train_est_filename, "rb") as f:
        load_res = pickle.load(f)
    if "Z" in load_res.keys(): # EM
        Z = load_res["Z"]
    elif "estimates" in load_res.keys(): # gradient ascent
        Z = load_res["estimates"]["Z"]

    bins_centers = test_filtering_res["bins_centers"]
    xnn = test_filtering_res["xnn"]
    Pnn = test_filtering_res["Pnn"]
    o_xnn, _ = ssm.neural_latents.utils.ortogonalizeMeansAndCovs(
        means=xnn, covs=Pnn, Z=Z)

    load_res = np.load(hmm_params_filename)
    state_labels = load_res["states_labels"]
    A = load_res["A"]
    means = load_res["means"]
    covs = load_res["covs"]

    K = means.shape[1]
    Pi = np.ones(K) * 1.0 / K # use non-informative prior for initial state
    p = hmmUtils.get_gaussian_probabilities(x=o_xnn.squeeze(), means=means,
                                            covs=covs)
    most_prob_states_seq = hmm.inference.viterbi(Pi=Pi, p=p, A=A)

    np.savez(most_prob_states_seq_filename,
             state_labels=state_labels,
             bins_centers=bins_centers,
             most_prob_states_seq=most_prob_states_seq)

    print(f"Most probable states saved to {most_prob_states_seq_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

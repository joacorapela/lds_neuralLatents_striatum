import sys
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd

import ssm.neural_latents.utils
import hmm.learning
import hmmUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtering_res_number", type=int,
                        help="number corresponding to filtered results filename",
                        default=42904130)
    parser.add_argument("--start_time_sec", help="start time to plot (sec)",
                        type=float, default=4045.0)
                        # type=float, default=7474.0)
                        # type=float, default=10474.0)
                        # type=float, default=5512.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        type=float, default=300.0)
                        # type=float, default=1000.0)
                        # type=float, default=120.0)
    parser.add_argument("--transitions_times_padding",
                        help="padding (sec) used in transitions times so that every bin center can be labeled",
                        type=float, default=10.0)
    parser.add_argument("--port_label_col_name",
                        help="column name for port label",
                        type=str, default="Start_Port")
    parser.add_argument("--port_enter_times_col_name",
                        help="column name for port enter (ephys) time",
                        type=str, default="P1_IN_Ephys_TS")
    parser.add_argument("--port_exit_times_col_name",
                        help="column name for port exit (ephys) time",
                        type=str, default="P1_OUT_Ephys_TS")
    parser.add_argument("--transitions_data_filename",
                        help="transition data filename", type=str,
                        default="/ceph/sjones/projects/sequence_squad/organised_data/animals/EJT178_implant1/recording6_29-03-2022/behav_sync/2_task/Transition_data_sync.csv")
                        # default="/nfs/gatsbystor/rapela/work/ucl/gatsby-swc/gatsby/svGPFA/repos/projects/svGPFA_striatum/data/Transition_data_sync.csv")
    parser.add_argument("--filtering_res_filenames_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_filtered.{:s}")
                        # default="../../results/EJT178_implant1/recording1_15-03-2022/{:08d}_filtered.{:s}")
    parser.add_argument("--hmm_params_filename_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_params_from{:.02f}_to{:.02f}.npz")

    args = parser.parse_args()

    filtering_res_number = args.filtering_res_number
    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    transitions_times_padding = args.transitions_times_padding
    port_label_col_name = args.port_label_col_name
    port_enter_times_col_name = args.port_enter_times_col_name
    port_exit_times_col_name = args.port_exit_times_col_name
    transitions_data_filename = args.transitions_data_filename
    filtering_res_filenames_pattern = \
        args.filtering_res_filenames_pattern
    hmm_params_filename = args.hmm_params_filename_pattern.format(
        filtering_res_number, start_time_sec, end_time_sec)

    transitions_data = pd.read_csv(transitions_data_filename)

    in_range_bool = np.logical_and(
        start_time_sec-transitions_times_padding<=transitions_data[port_enter_times_col_name],
        transitions_data[port_exit_times_col_name]<end_time_sec+transitions_times_padding)
    transitions_data = transitions_data[in_range_bool]

    filtering_res_filename = \
        filtering_res_filenames_pattern.format(filtering_res_number, "pickle")
    with open(filtering_res_filename, "rb") as f:
        filtering_res = pickle.load(f)

    bins_centers = filtering_res["bins_centers"]
    xnn = filtering_res["xnn"]
    Pnn = filtering_res["Pnn"]

    est_metadata_filename = filtering_res_filenames_pattern.format(filtering_res_number, "ini")
    est_metadata = configparser.ConfigParser()
    est_metadata.read(est_metadata_filename)
    est_res_num = int(est_metadata["params"]["est_res_num"])
    est_filename_pattern = est_metadata["params"]["est_filename_pattern"]

    est_filename = est_filename_pattern.format(est_res_num, "pickle")
    with open(est_filename, "rb") as f:
        est_res = pickle.load(f)
    Z = est_res["Z"]

    means, _ = ssm.neural_latents.utils.ortogonalizeMeansAndCovs(
        means=xnn, covs=Pnn, Z=Z)

    first_index = np.where(start_time_sec <= bins_centers)[0][0]
    last_index = np.where(bins_centers < end_time_sec)[0][-1]
    to_use_slice = slice(first_index, last_index)
    bins_centers_to_use = bins_centers[to_use_slice]
    xnn_to_use = means[:, :, to_use_slice]
    port_labels = transitions_data[port_label_col_name].to_numpy()
    enter_times = transitions_data[port_enter_times_col_name].to_numpy()
    exit_times = transitions_data[port_exit_times_col_name].to_numpy()
    states_labels, states_seq_to_use = hmmUtils.getFilteredBehavioralStates(
        bins_centers=bins_centers_to_use,
        port_labels=port_labels,
        enter_times=enter_times,
        exit_times=exit_times)
    Pi, A, means, covs = hmm.learning.supervisedLearningGaussianObservations(
        states_seq=np.array(states_seq_to_use),
        observations_seq=xnn_to_use.squeeze())

    np.savez(hmm_params_filename, states_labels=states_labels, Pi=Pi, A=A,
             bins_centers=bins_centers_to_use, means=means, covs=covs)
    print(f"HMM params saved to {hmm_params_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

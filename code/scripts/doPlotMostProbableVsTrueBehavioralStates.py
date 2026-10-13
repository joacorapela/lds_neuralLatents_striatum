
import sys
import argparse
import configparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go


import hmmUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_start_time_sec", help="train start time to plot (sec)",
                        type=float, default=4045.0)
    parser.add_argument("--train_duration_sec", help="train_duration to plot (sec)",
                        type=float, default=300.0)
    parser.add_argument("--test_start_time_sec", help="test start time to plot (sec)",
                        type=float, default=6000.0)
                        # type=float, default=4045.0)
    parser.add_argument("--test_duration_sec", help="test_duration to plot (sec)",
                        # type=float, default=300.0)
                        type=float, default=900.0)
    parser.add_argument("--train_filtering_res_number", type=int,
                        help="number corresponding to filtered results filename used to learn the params hmm",
                        default=42904130)
    parser.add_argument("--test_filtering_res_number", type=int,
                        help="number corresponding to filtered results filename used to test the hmm model",
                        # default=42904130)
                        # default=12157029)
                        default=21792914)
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
    parser.add_argument("--most_prob_states_seq_filename_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/train{:08d}_test{:08d}_hmm_most_prob_state_seq_trainFrom{:.02f}_trainTo{:.02f}_testFrom{:.02f}_testTo{:.02f}.npz")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/train{:08d}_test{:08d}_true_vs_inferred_hmm_most_prob_state_seq_trainFrom{:.02f}_trainTo{:.02f}_testFrom{:.02f}_testTo{:.02f}.{:s}")

    args = parser.parse_args()

    train_start_time_sec = args.train_start_time_sec
    test_start_time_sec = args.test_start_time_sec
    train_end_time_sec = args.train_start_time_sec + args.train_duration_sec
    test_end_time_sec = args.test_start_time_sec + args.test_duration_sec
    train_filtering_res_number = args.train_filtering_res_number
    test_filtering_res_number = args.test_filtering_res_number
    port_label_col_name = args.port_label_col_name
    port_enter_times_col_name = args.port_enter_times_col_name
    port_exit_times_col_name = args.port_exit_times_col_name
    transitions_data_filename = args.transitions_data_filename
    most_prob_states_seq_filename = args.most_prob_states_seq_filename_pattern.format(
        train_filtering_res_number, test_filtering_res_number,
        train_start_time_sec, train_end_time_sec,
        test_start_time_sec, test_end_time_sec)
    fig_filename_pattern = args.fig_filename_pattern

    load_res = np.load(most_prob_states_seq_filename)
    bins_centers_estimated = load_res["bins_centers"]
    most_prob_states_seq = load_res["most_prob_states_seq"]

    transitions_data = pd.read_csv(transitions_data_filename)

    in_range_bool = np.logical_and(
        test_start_time_sec<=transitions_data[port_enter_times_col_name],
        transitions_data[port_exit_times_col_name]<test_end_time_sec)
    transitions_data = transitions_data[in_range_bool]

    first_index = np.where(test_start_time_sec <= bins_centers_estimated)[0][0]
    last_index = np.where(bins_centers_estimated < test_end_time_sec)[0][-1]
    to_use_slice = slice(first_index, last_index)
    port_labels = transitions_data[port_label_col_name].to_numpy()
    enter_times = transitions_data[port_enter_times_col_name].to_numpy()
    exit_times = transitions_data[port_exit_times_col_name].to_numpy()

    states_labels_true, states_seq_true = hmmUtils.getFilteredBehavioralStates(
        bins_centers=bins_centers_estimated,
        port_labels=port_labels,
        enter_times=enter_times,
        exit_times=exit_times)

    fig = go.Figure()
    trace = go.Scatter(x=bins_centers_estimated, y=states_seq_true,
                       mode="lines+markers", name="True")
    fig.add_trace(trace)
    trace = go.Scatter(x=bins_centers_estimated, y=most_prob_states_seq,
                       mode="lines+markers", name="Estimated")
    fig.add_trace(trace)
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title="Inferred State Number")
    html_filename = fig_filename_pattern.format(train_filtering_res_number,
                                                test_filtering_res_number,
                                                train_start_time_sec, train_end_time_sec,
                                                test_start_time_sec, test_end_time_sec,
                                                "html")
    fig.write_html(html_filename)
    png_filename = fig_filename_pattern.format(train_filtering_res_number,
                                               test_filtering_res_number,
                                               train_start_time_sec, train_end_time_sec,
                                               test_start_time_sec, test_end_time_sec,
                                               "png")
    fig.write_image(png_filename)

    print(f"figure saved to {html_filename}")

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

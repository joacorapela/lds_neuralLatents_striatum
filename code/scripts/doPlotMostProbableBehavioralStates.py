
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
                        type=float, default=0.0)
                        # type=float, default=6000.0)
                        # type=float, default=4045.0)
    parser.add_argument("--test_duration_sec", help="test_duration to plot (sec)",
                        type=float, default=1000.0)
                        # type=float, default=1454.0)
                        # type=float, default=300.0)
    parser.add_argument("--train_filtering_res_number", type=int,
                        help="number corresponding to filtered results filename used to learn the params hmm",
                        default=42904130)
    parser.add_argument("--test_filtering_res_number", type=int,
                        help="number corresponding to filtered results filename used to test the hmm model",
                        default=37772953)
                        # default=21792914)
                        # default=42904130)
    parser.add_argument("--most_prob_states_seq_filename_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/train{:08d}_test{:08d}_hmm_most_prob_state_seq_trainFrom{:.02f}_trainTo{:.02f}_testFrom{:.02f}_testTo{:.02f}.npz")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/train{:08d}_test{:08d}_inferred_hmm_most_prob_state_seq_trainFrom{:.02f}_trainTo{:.02f}_testFrom{:.02f}_testTo{:.02f}.{:s}")

    args = parser.parse_args()

    train_start_time_sec = args.train_start_time_sec
    test_start_time_sec = args.test_start_time_sec
    train_end_time_sec = args.train_start_time_sec + args.train_duration_sec
    test_end_time_sec = args.test_start_time_sec + args.test_duration_sec
    train_filtering_res_number = args.train_filtering_res_number
    test_filtering_res_number = args.test_filtering_res_number

    most_prob_states_seq_filename = args.most_prob_states_seq_filename_pattern.format(
        train_filtering_res_number, test_filtering_res_number,
        train_start_time_sec, train_end_time_sec,
        test_start_time_sec, test_end_time_sec)
    fig_filename_pattern = args.fig_filename_pattern

    load_res = np.load(most_prob_states_seq_filename)
    bins_centers_estimated = load_res["bins_centers"]
    most_prob_states_seq = load_res["most_prob_states_seq"]

    fig = go.Figure()
    trace = go.Scatter(x=bins_centers_estimated, y=most_prob_states_seq,
                       mode="lines+markers")
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

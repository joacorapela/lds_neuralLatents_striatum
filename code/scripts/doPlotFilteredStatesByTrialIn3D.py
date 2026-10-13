import sys
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import ssm.plotting
import ssm.neural_latents.plotting
import striatumUtils
import plotUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time to plot (sec)",
                        type=float, default=4045.0)
                        # type=float, default=4045.0)
                        # type=float, default=7474.0)
                        # type=float, default=10474.0)
                        # type=float, default=5512.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        type=float, default=300.0)
                        # type=float, default=1000.0)
                        # type=float, default=120.0)
    parser.add_argument("--before_trial_pad",
                        help="pad (in seconds) to be added at begining of "
                        "trial before the beginning of the sequence",
                        type=float, default=0.2)
    parser.add_argument("--after_trial_pad",
                        help="pad (in seconds) to be added at end of "
                        "trial after the end of the sequence",
                        type=float, default=0.2)
    parser.add_argument("--latents_to_plot", help="latents to plot to plot",
                        type=str, default="[0,1,2]")
    parser.add_argument("--skip_orthonormalization",
                        help="set to skip the orthonormalization of the states",
                        action="store_true")
    parser.add_argument("--port_in_ephys_ts_col_name",
                        help="column name for port_in_ephys time stamp",
                        type=str, default="P1_IN_Ephys_TS")
    parser.add_argument("--port_out_ephys_ts_col_name",
                        help="column name for port_out_ephys time stamp",
                        type=str, default="P1_OUT_Ephys_TS")
    parser.add_argument("--ports_to_plot",
                        help="ports to plot", type=str,
                        default="1,2,3,4,5,6,7")
    parser.add_argument("--ports_linetypes",
                        help="linetypes for ports", type=str,
                        default="solid,solid,solid,solid,solid,solid,solid")
    parser.add_argument("--ports_colors",
                        help="colors for ports", type=str,
                        default="green,red,cyan,yellow,purple,blue,magenta")
    parser.add_argument("--transitions_data_filename",
                        help="transitions data filename", type=str,
                        default="/ceph/sjones/projects/sequence_squad/organised_data/animals/EJT178_implant1/recording6_29-03-2022/behav_sync/2_task/Transition_data_sync.csv")
                        # default="/nfs/gatsbystor/rapela/work/ucl/gatsby-swc/gatsby/svGPFA/repos/projects/svGPFA_striatum/data/Transition_data_sync.csv")
    parser.add_argument("--filtering_res_number", type=int,
                        help="number corresponding to filtered results filename",
                        default=42904130)
                        # default=51188144)
                        # default=83080162)
                        # default=85354126)
                        # default=69943603)
    parser.add_argument("--filtering_res_filenames_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_filtered.{:s}")
                        # default="../../results/EJT178_implant1/recording1_15-03-2022/{:08d}_filtered.{:s}")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_state_filtered_from{:.02f}_to{:.02f}_latents_{:d}_{:d}_{:d}.{:s}")
                        # default="../../figures/EJT178_implant1/recording1_15-03-2022/{:08d}_state_filtered_from{:.02f}_to{:.02f}.{:s}")

    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    before_trial_pad = args.before_trial_pad
    after_trial_pad = args.after_trial_pad
    latents_to_plot = [int(str) for str in args.latents_to_plot[1:-1].split(",")]
    skip_orthonormalization = args.skip_orthonormalization
    port_in_ephys_ts_col_name = args.port_in_ephys_ts_col_name
    port_out_ephys_ts_col_name = args.port_out_ephys_ts_col_name
    ports_to_plot = [int(port_str) for port_str in args.ports_to_plot.split(",")]
    ports_linetypes_str = args.ports_linetypes.split(",")
    ports_colors_str = args.ports_colors.split(",")
    transitions_data_filename = args.transitions_data_filename
    filtering_res_number = args.filtering_res_number
    filtering_res_filenames_pattern = \
        args.filtering_res_filenames_pattern
    fig_filename_pattern = args.fig_filename_pattern

    ports_linetypes = dict(zip(ports_to_plot, ports_linetypes_str))
    ports_colors = dict(zip(ports_to_plot, ports_colors_str))

    transitions_data = pd.read_csv(transitions_data_filename)

    # events_df = plotUtils.build_events_df(
    #     start_time_sec=start_time_sec,
    #     end_time_sec=end_time_sec,
    #     transitions_data=transitions_data,
    #     ports_linetypes=ports_linetypes,
    #     ports_colors=ports_colors)

    filtering_res_filename = \
        filtering_res_filenames_pattern.format(filtering_res_number, "pickle")
    with open(filtering_res_filename, "rb") as f:
        filtering_res = pickle.load(f)

    filtered_metadata_filename = \
        filtering_res_filenames_pattern.format(filtering_res_number, "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)
    est_filename = filtered_metadata["params"]["est_filename_pattern"].format(int(filtered_metadata["params"]["est_res_num"]), "pickle")
    with open(est_filename, "rb") as f:
        load_res = pickle.load(f)
    if "Z" in load_res.keys(): # EM
        Z = load_res["Z"]
    elif "estimates" in load_res.keys(): # gradient ascent
        Z = load_res["estimates"]["Z"]

    bins_centers = filtering_res["bins_centers"]
    first_index = np.where(start_time_sec <= bins_centers)[0][0]
    last_index = np.where(bins_centers < end_time_sec)[0][-1]
    to_plot_slice = slice(first_index, last_index)
    bins_centers_to_plot = bins_centers[to_plot_slice]
    means_to_plot = filtering_res["xnn"][:,:,to_plot_slice]
    covs_to_plot = filtering_res["Pnn"][:,:,to_plot_slice]

    in_range_bool = np.logical_and(
        start_time_sec<=transitions_data[port_in_ephys_ts_col_name],
        transitions_data[port_in_ephys_ts_col_name]<end_time_sec)
    transitions_data = transitions_data[in_range_bool]

    fig = go.Figure()
    trials_ids = transitions_data["Trial_id"].unique()
    n_trials = len(trials_ids)
    trials_ids_max = trials_ids.max()
    trials_start_times = [None for r in range(n_trials)]
    trials_end_times = [None for r in range(n_trials)]
    trials_times = [None for r in range(n_trials)]
    epoch_times = [None for r in range(n_trials)]
    for r in range(n_trials):
        trial_id = trials_ids[r]
        print(f"Processing trial {trials_ids[r]} ({trials_ids_max})")

        trial_transitions_data = transitions_data[transitions_data["Trial_id"]==trial_id]
        first_2in_time = trial_transitions_data.iloc[0][port_in_ephys_ts_col_name]
        epoch_time = first_2in_time
        trial_start_time = first_2in_time - before_trial_pad
        epoch_times[r] = epoch_time
        last_7out_time = trial_transitions_data.iloc[-1][port_out_ephys_ts_col_name]
        trial_end_time = last_7out_time + after_trial_pad
        trials_start_times[r] = trial_start_time
        trials_end_times[r] = trial_end_time
        trials_times[r] = bins_centers[np.logical_and(
            trial_start_time <= bins_centers, bins_centers < trial_end_time)]

    marked_events_times, marked_events_colors, marked_events_markers, \
            marked_events_labels = plotUtils.buildMarkedEventsInfoFromTransitions(
                transitions_data=transitions_data, trials_ids=trials_ids)

    align_event_times = [0.0 for r in range(n_trials)]

    fig = ssm.neural_latents.plotting.get3DPlotLatentsAcrossTrials(
        trials_times=trials_times, means=means_to_plot, covs=covs_to_plot,
        bins_centers=bins_centers_to_plot, Z=Z, trials_ids=trials_ids,
        latents_to_plot=latents_to_plot,
        orthonormalize=not skip_orthonormalization,
        align_event_times=align_event_times,
        epoch_times=epoch_times,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        marked_events_labels=marked_events_labels,
    )

    fig.update_layout(
        title=f'Log-Likelihood: {filtering_res["logLike"].squeeze()}')
    fig.write_image(fig_filename_pattern.format(filtering_res_number,
                                                start_time_sec, end_time_sec,
                                                latents_to_plot[0],
                                                latents_to_plot[1],
                                                latents_to_plot[2],
                                                "png"))
    html_filename = fig_filename_pattern.format(filtering_res_number,
                                                start_time_sec, end_time_sec,
                                                latents_to_plot[0],
                                                latents_to_plot[1],
                                                latents_to_plot[2],
                                                "html")
    fig.write_html(html_filename)
    print(f"Figure {html_filename} created")

    # fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

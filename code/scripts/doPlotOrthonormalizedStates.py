import sys
import pickle
import argparse
import configparser
import pynwb
import numpy as np

import ssm.neural_latents.plotting
import ssm.neural_latents.utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_type",
                        help="plot type: filtering or smoothing", type=str,
                        default="smoothing")
    parser.add_argument("--from_time", help="earliest time to plot",
                        type=float, default=-np.inf)
    parser.add_argument("--to_time", help="latest time to plot", type=float,
                        default=np.inf)
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--filepath_pattern", help="dandi filepath", type=str,
                        default="../../data/{:s}/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--events_names_to_plot",
                        help="names of events to plot", type=str,
                        default="start_time,target_on_time,go_cue_time,move_onset_time,stop_time")
    parser.add_argument("--events_linetypes_to_plot",
                        help="linetypes of events to plot", type=str,
                        default="dot,dash,dashdot,longdash,solid")
    parser.add_argument("--filtering_res_number", type=int,
                        help="filtered results filename number",
                        default=78991310)
#                         default=49497641)
#                         default=26118000)
#                         default=59816097)
    parser.add_argument("--variable", type=str, default="state",
                        help="variable to plot: state")
    parser.add_argument("--color_pattern_filtered", type=str,
                        default="rgba(255,0,0,{:f})",
                        help="color pattern for filtered data")
    parser.add_argument("--cb_alpha", type=float, default=0.3,
                        help="transparency factor for confidence bound")
    parser.add_argument("--results_filenames_pattern", type=str,
                        default="../../results/{:08d}_{:s}.{:s}",
                        help="results filename pattern")
    parser.add_argument("--smoothing_res_filenames_pattern", type=str,
                        default="../../results/{:08d}_smoothed.{:s}",
                        help="smoothing_res filename pattern")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_ortho_{:s}_{:s}_from{:.02f}_to{:.02f}.{:s}")

    args = parser.parse_args()

    plot_type = args.plot_type
    from_time = args.from_time
    to_time = args.to_time
    dandiset_ID = args.dandiset_ID
    filepath_pattern = args.filepath_pattern
    events_names_to_plot = args.events_names_to_plot.split(",")
    events_linetypes_to_plot = args.events_linetypes_to_plot.split(",")
    filtering_res_number = args.filtering_res_number
    variable = args.variable
    cb_alpha = args.cb_alpha
    results_filenames_pattern = \
        args.results_filenames_pattern
    fig_filename_pattern = args.fig_filename_pattern

    filepath = filepath_pattern.format(dandiset_ID)
    with pynwb.NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        trials_df = nwbfile.intervals["trials"].to_dataframe()

    filtering_res_filename = \
        results_filenames_pattern.format(filtering_res_number, "filtered",
                                         "pickle")
    with open(filtering_res_filename, "rb") as f:
        filtering_res = pickle.load(f)
    bin_centers = filtering_res["bin_centers"]
    log_like = filtering_res["logLike"].squeeze()

    first_index = np.where(bin_centers >= from_time)[0][0]
    last_index = np.where(bin_centers <= to_time)[0][-1]
    to_plot_slice = slice(first_index, last_index)
    bin_centers_to_plot = bin_centers[to_plot_slice]
    trials_df = trials_df[np.logical_and(
        trials_df['start_time'] >= from_time,
        trials_df['stop_time'] <= to_time)]

    if plot_type == "filtering":
        means_to_plot = filtering_res["xnn"][:, :, to_plot_slice]
        covs_to_plot = filtering_res["Pnn"][:, :, to_plot_slice]
    elif plot_type == "smoothing":
        smoothing_res_filename = \
            results_filenames_pattern.format(filtering_res_number,
                                             "smoothed", "pickle")
        with open(smoothing_res_filename, "rb") as f:
            smoothing_res = pickle.load(f)
        means_to_plot = smoothing_res["xnN"][:, :, to_plot_slice]
        covs_to_plot = smoothing_res["PnN"][:, :, to_plot_slice]
    else:
        raise RuntimeError(f"Invalid plot_type={plot_type}")

    filtered_metadata_filename = \
        results_filenames_pattern.format(filtering_res_number, "filtered",
                                         "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)
    est_res_num = int(filtered_metadata["params"]["est_res_num"])
    est_filename_pattern = filtered_metadata["params"]["est_filename_pattern"]
    est_filename = est_filename_pattern.format(est_res_num, "pickle")
    with open(est_filename, "rb") as f:
        optim_res = pickle.load(f)

    Z = optim_res["Z"]
    o_means_to_plot, o_covs_to_plot = \
        ssm.neural_latents.utils.ortogonalizeMeansAndCovs(means=means_to_plot,
                                                          covs=covs_to_plot,
                                                          Z=Z)

    if variable == "state":
        fig = ssm.neural_latents.plotting.plot_latents(
            means=o_means_to_plot,
            covs=o_covs_to_plot,
            bin_centers=bin_centers_to_plot,
            trials_df=trials_df,
            events_names_to_plot=events_names_to_plot,
            events_linetypes_to_plot=events_linetypes_to_plot,
            cb_alpha=cb_alpha, legend_pattern=f"{plot_type}_{{:d}}",
        )
    else:
        raise ValueError("variable={:s} is invalid.")

    fig.update_layout(
        title=f'Log-Likelihood: {log_like}')
    fig.write_image(fig_filename_pattern.format(filtering_res_number, variable,
                                                plot_type, from_time, to_time,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(filtering_res_number, variable,
                                               plot_type, from_time, to_time,
                                               "html"))
    fig.show()
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

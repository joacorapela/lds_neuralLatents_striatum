import sys
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd

import ssm.plotting
import ssm.neural_latents.plotting
import plotUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time to plot (sec)",
                        type=float, default=2240.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        type=float, default=120.0)
    parser.add_argument("--skip_orthogonalization",
                        help="set to skip the orthogonalization of the states",
                        action="store_true")
    parser.add_argument("--ports_to_plot",
                        help="ports to plot", type=str,
                        default="1,2,3,4,5,6,7")
    parser.add_argument("--ports_linetypes",
                        help="linetypes for ports", type=str,
                        default="solid,solid,solid,solid,solid,solid,solid")
    parser.add_argument("--ports_colors",
                        help="colors for ports", type=str,
                        default="blue,red,cyan,yellow,purple,green,magenta")
    parser.add_argument("--transition_data_filename",
                        help="transition data filename", type=str,
                        default="/nfs/gatsbystor/rapela/work/ucl/gatsby-swc/gatsby/svGPFA/repos/projects/svGPFA_striatum/data/Transition_data_sync.csv")
    parser.add_argument("--filtered_data_number", type=int,
                        help="number corresponding to filtered results filename",
                        default=91408320)
    parser.add_argument("--filtered_data_filenames_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}",
                        help="filtered_data filename pattern")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_state_filtered_from{:.02f}_to{:.02f}.{:s}")

    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    skip_orthogonalization = args.skip_orthogonalization
    ports_to_plot = [int(port_str) for port_str in args.ports_to_plot.split(",")]
    ports_linetypes_str = args.ports_linetypes.split(",")
    ports_colors_str = args.ports_colors.split(",")
    transition_data_filename = args.transition_data_filename
    filtered_data_number = args.filtered_data_number
    filtered_data_filenames_pattern = \
        args.filtered_data_filenames_pattern
    fig_filename_pattern = args.fig_filename_pattern

    ports_linetypes = dict(zip(ports_to_plot, ports_linetypes_str))
    ports_colors = dict(zip(ports_to_plot, ports_colors_str))

    transition_data = pd.read_csv(transition_data_filename)

    events_df = plotUtils.build_events_df(
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        transition_data=transition_data,
        ports_linetypes=ports_linetypes,
        ports_colors=ports_colors)

    filtered_data_filename = \
        filtered_data_filenames_pattern.format(filtered_data_number, "pickle")
    with open(filtered_data_filename, "rb") as f:
        filtered_data = pickle.load(f)

    filtered_metadata_filename = \
        filtered_data_filenames_pattern.format(filtered_data_number, "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)

    bins_centers = filtered_data["bins_centers"]
    first_index = np.where(bins_centers >= start_time_sec)[0][0]
    last_index = np.where(bins_centers <= end_time_sec)[0][-1]
    to_plot_slice = slice(first_index, last_index)
    bins_centers_to_plot = bins_centers[to_plot_slice]
    means_to_plot = filtered_data["xnn"][:,:,to_plot_slice]
    covs_to_plot = filtered_data["Pnn"][:,:,to_plot_slice]
    Z = filtered_data["Z"]

    fig = ssm.neural_latents.plotting.plot_latents(
        means=means_to_plot,
        covs=covs_to_plot,
        bins_centers=bins_centers_to_plot,
        orthogonalize=not skip_orthogonalization,
        Z=Z,
    )
    ssm.plotting.add_events_vlines(fig=fig, events_df=events_df)

    fig.update_layout(
        title=f'Log-Likelihood: {filtered_data["logLike"].squeeze()}')
    fig.write_image(fig_filename_pattern.format(filtered_data_number,
                                                start_time_sec, end_time_sec,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(filtered_data_number,
                                               start_time_sec, end_time_sec,
                                               "html"))
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

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
                        type=float, default=0.0)
                        # type=float, default=6000.0)
                        # type=float, default=7474.0)
                        # type=float, default=4345.0)
                        # type=float, default=4045.0)
                        # type=float, default=10474.0)
                        # type=float, default=5512.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        # type=float, default=1454.0)
                        # type=float, default=900.0)
                        # type=float, default=300.0)
                        type=float, default=1000.0)
                        # type=float, default=120.0)
    parser.add_argument("--skip_orthonormalization",
                        help="set to skip the orthonormalization of the states",
                        action="store_true")
    parser.add_argument("--ports_to_plot",
                        help="ports to plot", type=str,
                        default="1,2,3,4,5,6,7")
    parser.add_argument("--ports_linetypes",
                        help="linetypes for ports", type=str,
                        default="solid,solid,solid,solid,solid,solid,solid")
    parser.add_argument("--ports_colors",
                        help="colors for ports", type=str,
                        default="green,red,cyan,yellow,purple,blue,magenta")
    parser.add_argument("--transition_data_filename",
                        help="transition data filename", type=str,
                        default="/ceph/sjones/projects/sequence_squad/organised_data/animals/EJT178_implant1/recording6_29-03-2022/behav_sync/2_task/Transition_data_sync.csv")
    parser.add_argument("--filtering_res_number", type=int,
                        help="number corresponding to filtered results filename",
                        default=37772953)
                        # default=21792914)
                        # default=99907093)
                        # default=12157029)
                        # default=69943603)
                        # default=78975593)
                        # default=33121162)
                        # default=76606667)
                        # default=79497354)
                        # default=33368739)
                        # default=54945451)
                        # default=85354126)
                        # default=42904130)
                        # default=83080162)
                        # default=51188144)
                        # default=35171929)
                        # default=59077818)
                        # default=76873351)
                        # default=86836781)
                        # default=37634274)
                        # default=58340273)
                        # default=46183507)
    parser.add_argument("--filtering_res_filenames_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_filtered.{:s}")
                        # default="../../results/EJT178_implant1/recording1_15-03-2022/{:08d}_filtered.{:s}")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_state_filtered_from{:.02f}_to{:.02f}.{:s}")
                        # default="../../figures/EJT178_implant1/recording1_15-03-2022/{:08d}_state_filtered_from{:.02f}_to{:.02f}.{:s}")

    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    skip_orthonormalization = args.skip_orthonormalization
    ports_to_plot = [int(port_str) for port_str in args.ports_to_plot.split(",")]
    ports_linetypes_str = args.ports_linetypes.split(",")
    ports_colors_str = args.ports_colors.split(",")
    transition_data_filename = args.transition_data_filename
    filtering_res_number = args.filtering_res_number
    filtering_res_filenames_pattern = \
        args.filtering_res_filenames_pattern
    fig_filename_pattern = args.fig_filename_pattern

    ports_linetypes = dict(zip(ports_to_plot, ports_linetypes_str))
    ports_colors = dict(zip(ports_to_plot, ports_colors_str))

    transitions_data = pd.read_csv(transition_data_filename)

    events_df = plotUtils.build_events_df(
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        transitions_data=transitions_data,
        ports_linetypes=ports_linetypes,
        ports_colors=ports_colors)

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
    first_index = np.where(bins_centers >= start_time_sec)[0][0]
    last_index = np.where(bins_centers <= end_time_sec)[0][-1]
    to_plot_slice = slice(first_index, last_index)
    bins_centers_to_plot = bins_centers[to_plot_slice]
    means_to_plot = filtering_res["xnn"][:,:,to_plot_slice]
    covs_to_plot = filtering_res["Pnn"][:,:,to_plot_slice]

    fig = ssm.neural_latents.plotting.getPlotLatents(
        means=means_to_plot,
        covs=covs_to_plot,
        bins_centers=bins_centers_to_plot,
        orthonormalize=not skip_orthonormalization,
        Z=Z,
    )
    ssm.plotting.add_events_vlines(fig=fig, events_df=events_df)

    nro_samples = len(bins_centers)
    title = ('Log-Likelihood per sample: '
             f'{filtering_res["logLike"].squeeze() / nro_samples:.2E}')
    fig.update_layout(title=title)
    fig.write_image(fig_filename_pattern.format(filtering_res_number,
                                                start_time_sec, end_time_sec,
                                                "png"))
    html_filename = fig_filename_pattern.format(filtering_res_number,
                                                start_time_sec, end_time_sec,
                                                "html")
    fig.write_html(html_filename)
    print(f"Figure {html_filename} created")

    # fig.show()

    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)

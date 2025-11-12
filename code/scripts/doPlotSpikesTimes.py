
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import ssm.plotting
import plotUtils

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time to plot (sec)",
                        type=float, default=2240.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        type=float, default=360.0)
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
    parser.add_argument("--units_info_filename",
                        help="filename with units information", type=str,
                        default="/nfs/gatsbystor/rapela/work/ucl/gatsby-swc/gatsby/svGPFA/repos/projects/svGPFA_striatum/data/good_units_df.csv")
    parser.add_argument("--spikes_times_col_name",
                        help="column name for spikes times",
                        type=str, default="Spike_times")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", type=str,
                        default="../../figures/spike_times_start{:.2f}_end{:.2f}.{:s}")
    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    transition_data_filename = args.transition_data_filename
    units_info_filename = args.units_info_filename
    spikes_times_col_name = args.spikes_times_col_name
    ports_to_plot = [int(port_str) for port_str in args.ports_to_plot.split(",")]
    ports_linetypes_str = args.ports_linetypes.split(",")
    ports_colors_str = args.ports_colors.split(",")
    png_fig_filename = args.fig_filename_pattern.format(start_time_sec,
                                                        end_time_sec, "png")
    html_fig_filename = args.fig_filename_pattern.format(start_time_sec,
                                                         end_time_sec, "html")

    ports_linetypes = dict(zip(ports_to_plot, ports_linetypes_str))
    ports_colors = dict(zip(ports_to_plot, ports_colors_str))

    units_info_df = pd.read_csv(units_info_filename)
    transition_data = pd.read_csv(transition_data_filename)

    # n_clusters
    n_clusters = units_info_df.shape[0]

    # continuous spikes times
    continuous_spikes_times = [None for n in range(n_clusters)]
    for n in range(n_clusters):
        unit_spikes_times_str = units_info_df.iloc[n][spikes_times_col_name][1:-1].split(",")
        unit_spikes_times = np.array([float(unit_spike_times_str) for unit_spike_times_str in unit_spikes_times_str])
        continuous_spikes_times[n] = unit_spikes_times[np.logical_and(start_time_sec<=unit_spikes_times,
                                                                      unit_spikes_times<end_time_sec)]

    events_df = plotUtils.build_events_df(
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        transition_data=transition_data,
        ports_linetypes=ports_linetypes,
        ports_colors=ports_colors)

    fig = go.Figure()
    for n in range(n_clusters):
        trace = go.Scatter(x=continuous_spikes_times[n],
                           y=n*np.ones(len(continuous_spikes_times[n])),
                           mode="markers",
                           name=f'{n},{units_info_df.iloc[n]["Region"]}')
        fig.add_trace(trace)
    ssm.plotting.add_events_vlines(fig=fig, events_df=events_df)

    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title="Cluster Index")
    fig.write_image(png_fig_filename)
    fig.write_html(html_fig_filename)

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

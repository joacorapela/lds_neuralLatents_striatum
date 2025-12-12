import sys
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd
import plotly.express as px
import webcolors
import plotly.graph_objects as go

import ssm.inference
import ssm.plotting
import ssm.neural_latents.plotting
import plotUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", help="cluster to plot", type=int,
                        default=179)
                        # default=145)
    parser.add_argument("--horizon_sec", help="horizon (seconds)", type=float,
                        default=0.1)
    parser.add_argument("--filtering_res_number", type=int,
                        help="number corresponding to filtered results filename",
                        default=91973683)
                        # default=76873351)
                        # default=86836781)
                        # default=37634274)
                        # default=58340273)
                        # default=46183507)
    parser.add_argument("--ports_to_plot",
                        help="ports to plot", type=str,
                        default="1,2,3,4,5,6,7")
    parser.add_argument("--ports_linetypes",
                        help="linetypes for ports", type=str,
                        default="solid,solid,solid,solid,solid,solid,solid")
    parser.add_argument("--ports_colors",
                        help="colors for ports", type=str,
                        default="blue,red,cyan,yellow,purple,green,magenta")
    parser.add_argument("--cb_alpha", type=float,
                        help="transparency alpha for confidence bands",
                        default=0.3)
    parser.add_argument("--filtering_res_filenames_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}",
                        help="filtering_res filename pattern")
    parser.add_argument("--transition_data_filename",
                        help="transition data filename", type=str,
                        default="/nfs/gatsbystor/rapela/work/ucl/gatsby-swc/gatsby/svGPFA/repos/projects/svGPFA_striatum/data/Transition_data_sync.csv")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_obs_forecasts_horizon{:.03f}_cluster{:d}_from{:.02f}_to{:.02f}.{:s}")

    args = parser.parse_args()

    cluster = args.cluster
    horizon_sec = args.horizon_sec
    filtering_res_number = args.filtering_res_number
    ports_to_plot = [int(port_str) for port_str in args.ports_to_plot.split(",")]
    ports_linetypes_str = args.ports_linetypes.split(",")
    ports_colors_str = args.ports_colors.split(",")
    cb_alpha = args.cb_alpha
    filtering_res_filenames_pattern = \
        args.filtering_res_filenames_pattern
    transition_data_filename = args.transition_data_filename
    fig_filename_pattern = args.fig_filename_pattern

    ports_linetypes = dict(zip(ports_to_plot, ports_linetypes_str))
    ports_colors = dict(zip(ports_to_plot, ports_colors_str))

    transition_data = pd.read_csv(transition_data_filename)

    # get binned spikes
    filtered_metadata_filename = \
        filtering_res_filenames_pattern.format(filtering_res_number, "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)
    est_res_num = int(filtered_metadata["params"]["est_res_num"])
    est_filename_pattern = filtered_metadata["params"]["est_filename_pattern"]
    est_results_filename = est_filename_pattern.format(est_res_num, "pickle")
    start_time_sec = float(filtered_metadata["params"]["start_time_sec"])
    end_time_sec = float(filtered_metadata["params"]["end_time_sec"])

    with open(est_results_filename, "rb") as f:
        est_res = pickle.load(f)
    m0 = est_res["m0"]
    V0 = est_res["V0"]
    u = est_res["u"]
    B = est_res["B"]
    Q = est_res["Q"]
    a = est_res["a"]
    Z = est_res["Z"]
    R = est_res["R"]

    est_metadata_filename = est_filename_pattern.format(est_res_num, "ini")
    est_metadata = configparser.ConfigParser()
    est_metadata.read(est_metadata_filename)
    binned_spikes_filename = est_metadata["params"]["binned_spikes_filename"]
    clusters_indices_filename = est_metadata["params"]["clustersindices_filename"]
    clusters_indices = pd.read_csv(clusters_indices_filename, header=None, index_col=False).squeeze()

    binned_res = np.load(binned_spikes_filename)
    y = binned_res["binned_spikes"]
    bins_centers = binned_res["bins_centers"]

    s_rate = 1.0 / np.median(np.diff(bins_centers))
    h = int(np.round(horizon_sec * s_rate))

    valid_bins_mask = np.logical_and(start_time_sec<=bins_centers,
                                     bins_centers<end_time_sec)
    bins_centers = bins_centers[valid_bins_mask]
    y = y[clusters_indices, :]
    y = y[:, valid_bins_mask]

    filtering_res_filename = \
        filtering_res_filenames_pattern.format(filtering_res_number, "pickle")
    with open(filtering_res_filename, "rb") as f:
        filtering_res = pickle.load(f)
    xnn = filtering_res["xnn"]
    Pnn = filtering_res["Pnn"]

    # forecast observations
    forecast_mean, forecast_cov = ssm.inference.lds_observations_forecast(
        xnn=xnn, Pnn=Pnn, u=u, B=B, Q=Q, m0=m0, V0=V0, a=a, Z=Z, R=R, h=h)
    forecast_std = np.sqrt(np.diagonal(forecast_cov, axis1=0, axis2=1)).T

    # plot
    N = len(bins_centers)
    fig = go.Figure()
    events_df = plotUtils.build_events_df(
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        transition_data=transition_data,
        ports_linetypes=ports_linetypes,
        ports_colors=ports_colors)
    ssm.plotting.add_events_vlines(fig=fig, events_df=events_df)

    n = np.where(clusters_indices==cluster)[0].item()
    y_n = y[n,:]
    pred_mean_n = forecast_mean[n, :].squeeze()
    pred_std_n = forecast_std[n, :].squeeze()
    pred_mean_n_lower = pred_mean_n - 1.96 * pred_std_n
    pred_mean_n_upper = pred_mean_n + 1.96 * pred_std_n

    x = bins_centers[h:]
    pred_mean_n = pred_mean_n[:-h]
    pred_mean_n_lower = pred_mean_n_lower[:-h]
    pred_mean_n_upper = pred_mean_n_upper[:-h]

    trace = go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([pred_mean_n_upper, pred_mean_n_lower[::-1]]),
        fill="toself",
        fillcolor="rgba(0,255,0,0.3)",
        line=dict(color="rgba(0,255,0,0.0)"),
        showlegend=False,
        legendgroup=f"cluster{clusters_indices[n]}",
    )
    fig.add_trace(trace)

    trace = go.Scatter(x=bins_centers, y=y_n,
                       name="spike rate",
                       mode="lines+markers",
                       marker={"color": "rgba(255,0,0,1.0)",
                               "symbol": "square-open"})
    fig.add_trace(trace)

    trace = go.Scatter(x=x, y=pred_mean_n,
                       name="forecasting",
                       mode="lines+markers",
                       marker={"color": "rgba(0,255,0,1.0)", "symbol": "circle-open"},
                       legendgroup=f"cluster{clusters_indices[n]}")
    fig.add_trace(trace)

    fig.update_layout(title=f"Horizon: {h/s_rate:.03f} sec ({h} samples), Cluster: {cluster}")
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title="Firing Rate")

    fig.write_image(fig_filename_pattern.format(filtering_res_number,
                                                h/s_rate,
                                                cluster, start_time_sec,
                                                end_time_sec, "png"))
    fig.write_html(fig_filename_pattern.format(filtering_res_number,
                                               h/s_rate,
                                               cluster, start_time_sec,
                                               end_time_sec, "html"))
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

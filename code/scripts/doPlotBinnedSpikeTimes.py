
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_size", help="bin size (secs)", type=float,
                        default=0.02)
    parser.add_argument("--skip_log_transform",
                        help="log transform spike counts", action="store_true")
    parser.add_argument("--start_time_sec", help="start time to plot (sec)",
                        type=float, default=5512.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        type=float, default=120.0)
    parser.add_argument("--clustersIndices_filename",
                        help="filename with clusterIndices used for estimation", type=str,
                        default="../../metadata/clustersIndices_124_223.ini")
    parser.add_argument("--save_filename_pattern", help="save filename pattern", type=str,
                        default="../../results/binned_spikes_binSize{:.2f}_skipSqrtTrans{:d}.npz")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", type=str,
                        default="../../figures/binned_spikes__binSize{:.2f}_skipLogTrans{:d}.{{:s}}")
    args = parser.parse_args()

    bin_size = args.bin_size
    skip_log_transform = args.skip_log_transform
    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    clustersIndices_filename = args.clustersIndices_filename
    save_filename = args.save_filename_pattern.format(bin_size,
        skip_log_transform)
    fig_filename_pattern = args.fig_filename_pattern.format(bin_size,
        skip_log_transform)

    load_res = np.load(save_filename)

    spike_counts = load_res["binned_spikes"]
    bins_centers = load_res["bins_centers"]

    clustersIndices = pd.read_csv(clustersIndices_filename, header=None, index_col=False).squeeze()
    spike_counts = spike_counts[clustersIndices, :]

    valid_bins_mask = np.logical_and(start_time_sec<=bins_centers,
                                     bins_centers<end_time_sec)
    bins_centers = bins_centers[valid_bins_mask]
    spike_counts = spike_counts[:, valid_bins_mask]

    breakpoint()

    fig = go.Figure()
    trace = go.Heatmap(x=bins_centers, y=clustersIndices, z=spike_counts)
    fig.add_trace(trace)
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title="Cluster Index")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()


if __name__ == "__main__":
    main(sys.argv)

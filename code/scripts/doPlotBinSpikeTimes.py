
import sys
import argparse
import numpy as np
import plotly.graph_objects as go

import utils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--bin_size", help="bin size (secs)", type=float,
                        default=0.02)
    parser.add_argument("--skip_log_transform",
                        help="log transform spike counts", action="store_true")
    parser.add_argument("--save_filename_pattern", help="save filename pattern", type=str,
                        default="../../results/binned_spikes_dandisetID{:s}_binSize{:.2f}_skipLogTrans{:d}.npz")
    args = parser.parse_args()

    dandiset_ID = args.dandiset_ID
    bin_size = args.bin_size
    skip_log_transform = args.skip_log_transform
    save_filename = args.save_filename_pattern.format(dandiset_ID, bin_size,
                                                      skip_log_transform)
    load_res = np.load(save_filename)

    spike_counts = load_res["binned_spikes"]

    fig = go.Figure()
    trace = go.Heatmap(z=spike_counts)
    fig.add_trace(trace)
    fig.update_xaxes(title="Bin Index")
    fig.update_yaxes(title="Neuron Index")
    fig.show()


if __name__ == "__main__":
    main(sys.argv)

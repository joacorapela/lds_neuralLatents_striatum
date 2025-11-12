
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import gcnu_common.utils.neural_data_analysis


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_size", help="bin size", type=float, default=0.02)
    parser.add_argument("--skip_sqrt_transform",
                        help="skip sqrt transform spike counts", action="store_true")
    parser.add_argument("--units_info_filename",
                        help="filename with units information", type=str,
                        default="/nfs/gatsbystor/rapela/work/ucl/gatsby-swc/gatsby/svGPFA/repos/projects/svGPFA_striatum/data/good_units_df.csv")
    parser.add_argument("--spikes_times_col_name",
                        help="column name for spikes times",
                        type=str, default="Spike_times")
    parser.add_argument("--save_filename_pattern", help="save filename pattern", type=str,
                        default="../../results/binned_spikes_binSize{:.2f}_skipSqrtTrans{:d}.npz")
    args = parser.parse_args()

    bin_size = args.bin_size
    skip_sqrt_transform = args.skip_sqrt_transform
    units_info_filename = args.units_info_filename
    spikes_times_col_name = args.spikes_times_col_name
    save_filename = args.save_filename_pattern.format(bin_size,
                                                      skip_sqrt_transform)

    units_info_df = pd.read_csv(units_info_filename)

    # n_clusters
    n_clusters = units_info_df.shape[0]

    # continuous spikes times
    continuous_spikes_times = [None for n in range(n_clusters)]
    for n in range(n_clusters):
        unit_spikes_times_str = units_info_df.iloc[n][spikes_times_col_name][1:-1].split(",")
        unit_spikes_times = np.array([float(unit_spike_times_str) for unit_spike_times_str in unit_spikes_times_str])
        continuous_spikes_times[n] = unit_spikes_times

    t_min = min(min(st) for st in continuous_spikes_times if len(st) > 0)
    t_max = max(max(st) for st in continuous_spikes_times if len(st) > 0)
    bins_edges = np.arange(t_min, t_max, bin_size)
    bins_centers = (bins_edges[1:] + bins_edges[:-1])/2

    binned_spikes = np.empty(shape=(n_clusters, len(bins_centers)))
    for n in range(n_clusters):
        binned_spikes[n, :] = gcnu_common.utils.neural_data_analysis.binSpikesTimes(
            spikes_times=continuous_spikes_times[n], bins_edges=bins_edges,
            time_unit="sec")
    if not skip_sqrt_transform:
        binned_spikes = np.sqrt(binned_spikes + 0.5)

    np.savez(save_filename, bin_size=bin_size,
             slip_sqrt_transform=skip_sqrt_transform,
             binned_spikes=binned_spikes,
             bins_centers=bins_centers)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

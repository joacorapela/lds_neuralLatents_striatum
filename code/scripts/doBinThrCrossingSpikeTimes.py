
import sys
import argparse
import pickle
import numpy as np
import plotly.graph_objects as go

import gcnu_common.utils.neural_data_analysis


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time (sec)",
                        type=float, default=5512.0)
    parser.add_argument("--duration_sec", help="duration (sec)",
                        type=float, default=120.0)
                        # type=float, default=2.0)
    parser.add_argument("--bin_size", help="bin size", type=float, default=0.02)
    parser.add_argument("--skip_sqrt_transform",
                        help="skip sqrt transform spike counts", action="store_true")
    parser.add_argument("--spikes_filename_pattern", type=str,
                        help="spikes filename pattern",
                        default="../../results/thrCrossingSpikes_startTime{:.02f}_endTime{:.02f}.npy")
    parser.add_argument("--save_filename_pattern", help="save filename pattern", type=str,
                        default="../../results/binned_thrCrossingSpikes_startTime{:.02f}_endTime{:.02f}_binSize{:.2f}_skipSqrtTrans{:d}.npz")
    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    bin_size = args.bin_size
    skip_sqrt_transform = args.skip_sqrt_transform
    spikes_filename = args.spikes_filename_pattern.format(start_time_sec,
                                                          end_time_sec)
    save_filename = args.save_filename_pattern.format(start_time_sec,
                                                      end_time_sec,
                                                      bin_size,
                                                      skip_sqrt_transform)

    with open(spikes_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spike_times"]
    n_electrodes = len(spikes_times)

    t_min = min(min(st) for st in spikes_times if len(st) > 0)
    t_max = max(max(st) for st in spikes_times if len(st) > 0)
    bins_edges = np.arange(t_min, t_max, bin_size)
    bins_centers = (bins_edges[1:] + bins_edges[:-1])/2

    binned_spikes = np.empty(shape=(n_electrodes, len(bins_centers)))
    for n in range(n_electrodes):
        binned_spikes[n, :] = gcnu_common.utils.neural_data_analysis.binSpikesTimes(
            spikes_times=spikes_times[n], bins_edges=bins_edges,
            time_cluster="sec")
    if not skip_sqrt_transform:
        binned_spikes = np.sqrt(binned_spikes + 0.5)

    np.savez(save_filename, bin_size=bin_size,
             slip_sqrt_transform=skip_sqrt_transform,
             binned_spikes=binned_spikes,
             bins_centers=bins_centers)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

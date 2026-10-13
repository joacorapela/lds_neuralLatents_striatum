
import sys
import argparse
import configparser
import pickle
import numpy as np
import plotly.graph_objects as go

import gcnu_common.utils.neural_data_analysis


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time (sec)",
                        type=float, default=7474.0)
                        # type=float, default=4045.0)
                        # type=float, default=5512.0)
    parser.add_argument("--duration_sec", help="duration (sec)",
                        type=float, default=300.0)
                        # type=float, default=120.0)
                        # type=float, default=2.0)
    parser.add_argument("--bin_size_sec", help="bin size", type=float, default=0.2)
    parser.add_argument("--skip_sqrt_transform",
                        help="skip sqrt transform spike counts", action="store_true")
    parser.add_argument("--spikes_filename_pattern", type=str,
                        help="spikes filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/thrCrossingSpikes_startTime{:.02f}_duration{:.02f}.{{:s}}")
                        # default="../../results/EJT178_implant1/recording1_15-03-2022/thrCrossingSpikes_startTime{:.02f}_duration{:.02f}.{{:s}}")
    parser.add_argument("--results_filename_pattern", help="save filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/binned_thrCrossingSpikes_startTime{:.02f}_duration{:.02f}_binSize{:.2f}_skipSqrtTrans{:d}.{{:s}}")
                        # default="../../results/EJT178_implant1/recording1_15-03-2022/binned_thrCrossingSpikes_startTime{:.02f}_duration{:.02f}_binSize{:.2f}_skipSqrtTrans{:d}.{{:s}}")
    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    duration_sec = args.duration_sec
    bin_size_sec = args.bin_size_sec
    skip_sqrt_transform = args.skip_sqrt_transform
    spikes_filename_pattern = args.spikes_filename_pattern.format(
        start_time_sec, duration_sec)
    results_filename_pattern = args.results_filename_pattern.format(
        start_time_sec, duration_sec, bin_size_sec, skip_sqrt_transform)

    spikes_times_metadata_filename = \
        spikes_filename_pattern.format("ini")
    spikes_times_metadata = configparser.ConfigParser()
    spikes_times_metadata.read(spikes_times_metadata_filename)
    fs = int(spikes_times_metadata["params"]["fs"])

    spikes_filename = spikes_filename_pattern.format("pickle")
    with open(spikes_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_samples = load_res["spikes_times"]
    # convert spikes_samples to spikes_times
    spikes_times = [[float(spike_sample)/fs for spike_sample in a_spikes_samples]
                    for a_spikes_samples in spikes_samples]

    n_electrodes = len(spikes_times)

    t_min = min(min(st) for st in spikes_times if len(st) > 0)
    t_max = max(max(st) for st in spikes_times if len(st) > 0)
    bins_edges = np.arange(t_min, t_max, bin_size_sec)
    bins_centers = (bins_edges[1:] + bins_edges[:-1])/2

    binned_spikes = np.empty(shape=(n_electrodes, len(bins_centers)))
    for n in range(n_electrodes):
        binned_spikes[n, :] = gcnu_common.utils.neural_data_analysis.binSpikesTimes(
            spikes_times=spikes_times[n], bins_edges=bins_edges,
            time_cluster="sec")
    if not skip_sqrt_transform:
        binned_spikes = np.sqrt(binned_spikes + 0.5)

    results_filename = results_filename_pattern.format("npz")
    np.savez(results_filename, bins_centers=bins_centers,
             binned_spikes=binned_spikes)

    metadata_filename = results_filename_pattern.format("ini")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "start_time_sec": start_time_sec,
        "duration_sec": args.duration_sec,
        "bin_size_sec": bin_size_sec,
        "spikes_filename": spikes_filename,
        "results_filename": results_filename,
        "metadata_filename": metadata_filename,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

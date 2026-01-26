
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import open_ephys.analysis
import plotly.graph_objects as go

import spikeSortingUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time (sec)",
                        type=float, default=5512.0)
    parser.add_argument("--duration_sec", help="duration (sec)",
                        type=float, default=120.0)
                        # type=float, default=2.0)
    parser.add_argument("--spike_size_stds",
                        help="size of spike in standard deviation units",
                        type=float, default=5.0)
    parser.add_argument("--dead_time_ms",
                        help="dead time between two consecutive spikes (ms)",
                        type=float, default=0.3)
    parser.add_argument("--channel_to_plot",
                        help="channel to plot",
                        type=int, default=170)
    parser.add_argument("--plot_downsample_factor",
                        help="plot downsample factor",
                        type=int, default=10)
                        # type=float, default=2.0)
    parser.add_argument("--raw_ephys_data_dirname", type=str,
                        help="dirname containing the raw data",
                        default="/ceph/sjones/projects/sequence_squad/data/raw_neuropixel/OE_DATA/EJT178/290322/2022-03-29_13-55-06")
    parser.add_argument("--selected_channels_filename", type=str,
                        help="figure filename pattern",
                        default="../../data/channels0_383.csv")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/spikesFromThresholdCrossing_startTime{:.02f}_endTime{:.02f}_channel{:03d}.{{:s}}")
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../results/thrCrossingSpikes_startTime{:.02f}_endTime{:.02f}.npy")
    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    spike_size_stds = args.spike_size_stds
    dead_time_ms = args.dead_time_ms
    channel_to_plot = args.channel_to_plot
    plot_downsample_factor = args.plot_downsample_factor
    raw_ephys_data_dirname = args.raw_ephys_data_dirname
    selected_channels_filename = args.selected_channels_filename
    fig_filename_pattern = args.fig_filename_pattern.format(start_time_sec,
                                                            end_time_sec,
                                                            channel_to_plot)
    results_filename = args.results_filename_pattern.format(start_time_sec,
                                                            end_time_sec)

    selected_channels_df = pd.read_csv(selected_channels_filename, header=None)
    selected_channels = selected_channels_df.iloc[0].to_list()
    print(f"Loading and processing ephys data at {raw_ephys_data_dirname}")
    session_data = open_ephys.analysis.Session(raw_ephys_data_dirname)

    recording = session_data.recordnodes[0].recordings[0]
    recording.add_sync_line(1,102,"0",main=True)
    recording.add_sync_line(1,109,"0",main=False)

    recording.compute_global_timestamps()

    # Define the streams
    ap_stream = recording.continuous[0]

    ap_band_sample_rate = ap_stream.metadata["sample_rate"]
    ap_band_start_time = ap_stream.global_timestamps[0]
    if start_time_sec < ap_band_start_time:
        raise ValueError(f"start_time_sec={start_time_sec} should be >= than "
                         f"ap_band_start_time={ap_band_start_time}")
    ap_band_end_time = ap_stream.global_timestamps[-1]
    if end_time_sec > ap_band_end_time:
        raise ValueError(f"end_time_sec={end_time_sec} should be <= than "
                         f"ap_band_end_time={ap_band_end_time}")
    ap_band_start_sample = int((start_time_sec-ap_band_start_time) * ap_band_sample_rate)
    ap_band_end_sample = int((end_time_sec-ap_band_start_time) * ap_band_sample_rate)
    ap_band_timestamps = ap_stream.global_timestamps[
        ap_band_start_sample:ap_band_end_sample]

    spikes_times = [None] * len(selected_channels)

    for n, selected_channel in enumerate(selected_channels):
        start_time = time.time()
        ap_band_samples_muV = ap_stream.get_samples(
            start_sample_index=ap_band_start_sample,
            end_sample_index=ap_band_end_sample,
            selected_channels=[selected_channel],
        ).flatten()
        elapsed_time = time.time() - start_time
        print(f"Processing time for channel {selected_channel}: "
              f"{elapsed_time} sec")
        ap_band_samples_muV = ap_band_samples_muV - np.median(ap_band_samples_muV)

        # Calculate noise level (using the median to avoid spike bias)
        sigma_n = np.median(np.abs(ap_band_samples_muV) / 0.6745)
        threshold = -spike_size_stds * sigma_n  # Spikes are negative-going in extracellular recordings

        spike_indices = spikeSortingUtils.detect_mua_peaks(
            data=ap_band_samples_muV,
            threshold=threshold,
            dead_time_ms=dead_time_ms,
            fs=ap_band_sample_rate)

        spikes_times[n] = ap_band_timestamps[spike_indices]
        if n == channel_to_plot:
            ap_band_samples_muV_to_plot = ap_band_samples_muV
            threshold_to_plot = threshold

    results = {"spikes_times": spikes_times}
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Spike times saved to {results_filename}")

    # plotting
    fig = go.Figure()
    trace = go.Scatter(x=ap_band_timestamps[::plot_downsample_factor],
                       y=ap_band_samples_muV_to_plot[::plot_downsample_factor],
                       mode="markers+lines")
    fig.add_trace(trace)
    fig.add_hline(y=threshold_to_plot)
    fig.add_vline(x=spikes_times[channel_to_plot])
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title=r"$\text{Voltage}\;(\mu V)$")
    fig.update_layout(title=f"Electrode {channel_to_plot:03d}")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

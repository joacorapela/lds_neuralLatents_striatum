import sys
import argparse
import configparser
import pickle
import numpy as np
from scipy.signal import butter, sosfiltfilt
from open_ephys.analysis import Session


def extractSpikeTimesByThrCrossing(open_ephys_dir, start_time_sec,
                                   duration_sec, chunck_duration_sec,
                                   median_ch_spacing=20,
                                   refractory_sec=0.0015,
                                   threshold_multiplier=5,
                                   high_pass_order=3,
                                   high_pass_critical=300):
    # 1. Setup paths and load session
    session = Session(open_ephys_dir)
    recording = session.recordnodes[0].recordings[0]
    continuous = recording.continuous[0]

    # Metadata
    fs = continuous.metadata['sample_rate']
    n_channels = continuous.metadata['num_channels']
    total_samples = continuous.samples.shape[0]

    sample_start = int(start_time_sec * fs)
    n_samples_total = int(duration_sec * fs)
    n_samples_per_chunck = int(chunck_duration_sec * fs)

    # Subset of channels to estimate the median (e.g., every 20th channel)
    ref_chans = np.linspace(0, n_channels - 1, median_ch_spacing).astype(int)

    # 2. Parameters
    chunk_samples = n_samples_per_chunck
    refractory_samples = int(refractory_sec * fs) # 1.5ms

    # 3. High-pass filter
    sos = butter(N=high_pass_order, Wn=high_pass_critical, btype='hp', fs=fs,
                 output="sos")

    # 4. Initialize output: list of lists
    spikes_times = [[] for _ in range(n_channels)]

    print(f"Starting chunked detection. Total duration: {n_samples_total/fs:.2f}s")

    # 5. Main Loop: Iterate through Time Chunks
    for start in range(sample_start, sample_start+n_samples_total,
                       n_samples_per_chunck):
        end = start + n_samples_per_chunck

        # A. Load reference subset for this chunk to calculate Median
        # Shape will be (n_samples, n_channels/median_ch_spacing)
        ref_data = continuous.get_samples(start, end, selected_channels=ref_chans)
        chunk_median = np.median(ref_data, axis=1)

        # B. Process each channel within this time chunk
        for c in range(n_channels):
            # Load single channel for this chunk
            raw_chunk = continuous.get_samples(start, end, selected_channels=[c]).flatten()

            # Subtract common median (CMR)
            referenced_chunk = raw_chunk - chunk_median

            # Filter
            filtered_chunk = sosfiltfilt(sos, referenced_chunk)

            # Calculate robust threshold for this specific channel/chunk
            sigma_n = np.median(np.abs(filtered_chunk)) / 0.6745
            thr = -threshold_multiplier * sigma_n

            # Detect crossings
            crossings = np.where(filtered_chunk < thr)[0]

            if len(crossings) > 0:
                # Simple refractory period logic for the chunk
                current_spikes = [crossings[0].item()]
                for s in crossings[1:]:
                    if s - current_spikes[-1] > refractory_samples:
                        current_spikes.append(s.item())

                # Map chunk-indices back to global sample indices and store
                global_indices = [idx + start for idx in current_spikes]
                spikes_times[c].extend(global_indices)

        progress = (end - sample_start) / n_samples_total * 100
        print(f"Progress: {progress:.1f}% | Processed up to {(end-sample_start)/fs:.1f}s")

    print("Spike detection complete.")
    return spikes_times


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time to analyze (sec)",
                        type=float, default=5512.0)
    parser.add_argument("--duration_sec", help="duration to analyze (sec)",
                        type=float, default=120.0)
    parser.add_argument("--chunck_duration_sec", help="chunck duration (sec)",
                        type=float, default=1.0)
    parser.add_argument("--median_ch_spacing",
                        help=("separation between channels used to estimate "
                              "the channels median"),
                        type=int, default=20)
    parser.add_argument("--refractory_sec",
                        help="refractory period duration (sec)",
                        type=float, default=0.0015)
    parser.add_argument("--threshold_multiplier",
                        help=("voltages below threshold_multiplier * std of a "
                              "channel potential are considered spikes"),
                        type=float, default=5.0)
    parser.add_argument("--high_pass_order",
                        help="high-pass filter order",
                        type=int, default=3)
    parser.add_argument("--high_pass_critical",
                        help="high-pass filter critical value (Hz)",
                        type=float, default=300)
    parser.add_argument("--open_ephys_dir",
                        help="open ephy outputs directory name",
                        type=str,
                        default="/ceph/sjones/projects/sequence_squad/data/raw_neuropixel/OE_DATA/EJT178/150322/2022-03-15_13-55-36")
    parser.add_argument("--results_filename_pattern",
                        help="results filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording1_15-03-2022/thrCrossingSpikes_startTime{:.02f}_duration{:.02f}.{:s}")
    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    duration_sec = args.duration_sec
    chunck_duration_sec = args.chunck_duration_sec
    refractory_sec = args.refractory_sec
    median_ch_spacing = args.median_ch_spacing
    threshold_multiplier = args.threshold_multiplier
    high_pass_order = args.high_pass_order
    high_pass_critical=args.high_pass_critical
    open_ephys_dir = args.open_ephys_dir
    results_filename_pattern = args.results_filename_pattern

    results_filename = results_filename_pattern.format(start_time_sec,
                                                       duration_sec, "pickle")
    metadata_filename = results_filename_pattern.format(start_time_sec,
                                                        duration_sec, "ini")
    spikes_times = extractSpikeTimesByThrCrossing(
        open_ephys_dir=open_ephys_dir,
        start_time_sec=start_time_sec,
        duration_sec=duration_sec,
        chunck_duration_sec=chunck_duration_sec,
        median_ch_spacing=median_ch_spacing,
        refractory_sec=args.refractory_sec,
        threshold_multiplier=args.threshold_multiplier,
        high_pass_order=high_pass_order,
        high_pass_critical=high_pass_critical)

    results = {"spikes_times": spikes_times}
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved spike times to {results_filename}")

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "start_time_sec": start_time_sec,
        "duration_sec": args.duration_sec,
        "chunck_duration_sec": chunck_duration_sec,
        "refractory_sec": refractory_sec,
        "median_ch_spacing": median_ch_spacing,
        "threshold_multiplier": threshold_multiplier,
        "high_pass_order": high_pass_order,
        "high_pass_critical": high_pass_critical,
        "fs": fs,
        "open_ephys_dir": open_ephys_dir,
        "results_filename": results_filename,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

import sys
import pathlib
import os.path
import argparse
import numpy as np
import plotly.graph_objects as go


def plot_fig1a_heatmap(data_path, start_time_sec, duration_sec, fs, n_channels):
    path = pathlib.Path(data_path)

    sample_start = int(start_time_sec * fs)
    n_samples = int(duration_sec * fs)

    # 1. Load Pre-processed Data
    # Most KS3 runs use int16 for temp_wh.dat. Adjust to float32 if needed.
    dat_path = os.path.join(path, 'temp_wh.dat')
    bytes_per_val = 2 # int16

    # Correct Offset: Skip 'sample_start' timepoints for ALL channels
    offset = sample_start * n_channels * bytes_per_val

    with open(dat_path, 'rb') as f:
        f.seek(offset)
        # Read n_samples for all channels
        data = np.fromfile(f, dtype='int16', count=n_channels * n_samples)
        # Reshape to (Channels, Time) for the Heatmap
        data = data.reshape((n_samples, n_channels)).T 

    # 2. Create the Heatmap
    # Using 'RdBu_r' gives you the classic Red/Blue divergent look for spikes
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=start_time_sec+np.arange(n_samples) / fs, # Time axis
        y=np.arange(n_channels), # Electrode axis
        colorscale='RdBu_r',
        zmid=0, # Center the color scale at 0 (baseline)
        zmin=-np.std(data)*5, # Contrast stretch to see spikes over noise
        zmax=np.std(data)*5,
        colorbar=dict(title="Whitened V")
    ))

    fig.update_layout(
        title="Fig 1a Reproduction: Pre-processed Data Heatmap",
        xaxis_title="Time (sec)",
        yaxis_title="Electrode Number",
        width=900,
        height=600,
        template="plotly_white"
    )

    fig.show()
    breakpoint()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec", help="start time (sec)",
                        type=float, default=5584.0)
                        # type=float, default=5578.0)
    parser.add_argument("--duration_sec", help="duration (sec)",
                        type=float, default=2.0)
                        # type=float, default=120.0)
    parser.add_argument("--fs", help="sample rate",
                        type=int, default=30000)
    parser.add_argument("--n_channels", help="number of channels",
                        type=int, default=383)
    parser.add_argument("--kilosort3_dirname", help="kilosort3 directory name",
                        type=str,
                        default="/ceph/sjones/projects/sequence_squad/data/kilosort_output/EJT178/Recording6/2022-03-29_13-55-06/kilosort3")
    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    duration_sec = args.duration_sec
    fs = args.fs
    n_channels = args.n_channels
    kilosort3_diname = args.kilosort3_dirname

    plot_fig1a_heatmap(data_path=kilosort3_diname, start_time_sec=start_time_sec,
                       duration_sec=duration_sec, fs=fs, n_channels=n_channels)


if __name__ == "__main__":
    main(sys.argv)

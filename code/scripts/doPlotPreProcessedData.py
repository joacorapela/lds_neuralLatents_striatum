import sys
import os.path
import argparse
import pathlib
import numpy as np
import plotly.graph_objects as go


def reproduce_fig1a_actual(data_path, n_channels=383, sample_start=1000, n_samples=100, v_scale=5):
    path = pathlib.Path(data_path)

    # 1. Load the pre-processed (whitened/filtered) binary data
    # Kilosort saves this as an 'int16' or 'float32' binary file
    dat_path = os.path.join(path, 'temp_wh.dat')

    bytes_per_sample_all_channels = n_channels * np.dtype('int16').itemsize
    offset = sample_start * bytes_per_sample_all_channels

    with open(dat_path, 'rb') as f:
        f.seek(offset)
        data = np.fromfile(f, dtype='int16', count=n_channels * n_samples)
        # Convert to float for processing/plotting
        data = data.astype('float32') 
        data = data.reshape((n_samples, n_channels))

    # 2. Load channel positions
    ch_pos = np.load(os.path.join(path, 'channel_positions.npy'))

    # 3. Create the Plotly figure (B&W aesthetic)
    fig = go.Figure()
    t = np.arange(n_samples)

    for ch in range(n_channels):
        x_pos, y_pos = ch_pos[ch]

        # Plot the high-passed, unwhitened trace at its location
        # v_scale mimics the "zoom" in the paper to make spikes visible
        trace = data[:, ch] * v_scale

        fig.add_trace(go.Scattergl(
            x=t + x_pos,
            y=trace + y_pos,
            mode='lines',
            line=dict(color='black', width=0.5),
            hoverinfo='none',
            showlegend=False
        ))
        breakpoint()

    # 4. Scientific Paper Styling
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(title="μm", showgrid=False, linecolor='black', mirror=True),
        yaxis=dict(title="μm", showgrid=False, linecolor='black', mirror=True, scaleanchor="x"),
        width=600,
        height=900,
        title="Actual Fig 1a: High-Pass Filtered & Unwhitened Raw Data"
    )

    fig.show()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--kilosort3_dirname", help="kilosort3 directory name",
                        type=str,
                        default="/ceph/sjones/projects/sequence_squad/data/kilosort_output/EJT178/Recording6/2022-03-29_13-55-06/kilosort3")
    args = parser.parse_args()

    kilosort3_diname = args.kilosort3_dirname

    reproduce_fig1a_actual(data_path=kilosort3_diname)


if __name__ == "__main__":
    main(sys.argv)

import sys
import os.path
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def plot_template_plotly(data_path, cluster_id, n_channels=12, fs=30000):
    path = Path(data_path)

    # 1. Load data
    templates = np.load(os.path.join(path, 'templates.npy'))             # (n_templates, n_samples, n_channels)
    whitening_inv = np.load(os.path.join(path, 'whitening_mat_inv.npy')) # (n_channels, n_channels)
    channel_pos = np.load(os.path.join(path, 'channel_positions.npy'))    # (n_channels, 2)

    breakpoint()
    # 2. Unwhiten & identify peak
    # Templates are unit-norm; unwhitening restores physical shape
    unwhitened = np.dot(templates[cluster_id], whitening_inv)
    peak_ch = np.argmax(np.ptp(unwhitened, axis=0))

    # 3. Find nearest neighbors for spatial context
    dist = np.linalg.norm(channel_pos - channel_pos[peak_ch], axis=1)
    best_chans = np.argsort(dist)[:n_channels]

    # 4. Create Plotly Subplots
    fig = make_subplots(
        rows=1, cols=n_channels, 
        subplot_titles=[f"Ch {c}" for c in best_chans],
        shared_yaxes=True,
        horizontal_spacing=0.02
    )

    time_ms = (np.arange(unwhitened.shape[0]) / fs) * 1000 # Convert to ms

    for i, ch in enumerate(best_chans):
        is_peak = (ch == peak_ch)
        fig.add_trace(
            go.Scatter(
                x=time_ms, 
                y=unwhitened[:, ch],
                mode='lines',
                name=f"Channel {ch}",
                line=dict(color='red' if is_peak else 'black', width=2 if is_peak else 1),
                hovertemplate="Time: %{x:.2f}ms<br>Amp: %{y:.4f}<extra></extra>"
            ),
            row=1, col=i+1
        )

    # 5. Styling
    fig.update_layout(
        title=f"Kilosort Template {cluster_id} (Interactive) | Peak Channel: {peak_ch}",
        template="plotly_white",
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fig.update_xaxes(title_text="ms", row=1, col=int(n_channels/2))
    fig.show()


def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("--cluster_id", type=int, help="cluster ID", default=170)
	parser.add_argument("--kilosort_dirname", type=str,
                        help="Kilosort output dirname",
                        default="/ceph/sjones/projects/sequence_squad/data/kilosort_output/EJT178/Recording1/2022-03-15_13-55-36/kilosort3")
	args = parser.parse_args()

	kilosort_dirname = args.kilosort_dirname
	cluster_id = args.cluster_id

	plot_template_plotly(data_path=kilosort_dirname, cluster_id=cluster_id)


if __name__=="__main__":
	main(sys.argv)

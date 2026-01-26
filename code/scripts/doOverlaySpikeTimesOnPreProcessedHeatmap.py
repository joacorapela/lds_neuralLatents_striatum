import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def plot_spikes_on_heatmap(data_path, sample_start, n_samples,
                           n_channels=383, bytes_per_val=2):
    path = Path(data_path)

    # --- 1. Load Pre-processed Data ---
    offset = sample_start * n_channels * bytes_per_val
    
    with open(path / 'temp_wh.dat', 'rb') as f:
        f.seek(offset)
        data = np.fromfile(f, dtype='int16', count=n_channels * n_samples)
        data = data.reshape((n_samples, n_channels)).T

    # --- 2. Load Templates & Calculate Peak Channels ---
    # templates.npy shape is (n_templates, n_samples_in_template, n_channels)
    templates = np.load(path / 'templates.npy')
    
    # For each template, find which channel has the maximum peak-to-peak amplitude
    template_peak_chans = np.argmax(np.ptp(templates, axis=1), axis=1)

    # --- 3. Load Spikes ---
    spike_times = np.load(path / 'spike_times.npy').flatten()
    # Use spike_templates to get the original template index for each spike
    spike_templates = np.load(path / 'spike_templates.npy').flatten()

    # --- 4. Filter for our Time Window ---
    sample_end = sample_start + n_samples
    mask = (spike_times >= sample_start) & (spike_times < sample_end)
    
    window_spike_times = spike_times[mask]
    # window_spike_times = spike_times[mask] - sample_start
    # Get the peak channel for the template associated with each spike
    window_spike_chans = template_peak_chans[spike_templates[mask]]

    sample_start = 60000
    n_samples = 30000
    results_filename_pattern = "../../results/thrCrossingSpikes_startSample{:d}_endSample{:d}.pickle"
    results_filename = results_filename_pattern.format(sample_start, sample_start+n_samples-1)

    with open(results_filename, "rb") as f:
        load_res = pickle.load(f)
    thr_crossing_spikes_times = load_res["spikes_times"]

    # --- 5. Plotting ---
    x_axis_values = np.arange(sample_start, sample_start+n_samples) 
    y_axis_values = np.arange(n_channels) 
    fig = go.Figure()

    # The Heatmap
    fig.add_trace(go.Heatmap(
        x=x_axis_values,
        y=y_axis_values,
        z=data,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-np.std(data)*8,
        zmax=np.std(data)*8,
        showscale=False
    ))

    # The Spike Markers
    fig.add_trace(go.Scatter(
        x=window_spike_times,
        y=window_spike_chans,
        mode='markers',
        marker=dict(color='yellow', size=7, symbol='circle', line=dict(width=1, color='black')),
        # marker=dict(color='yellow', size=20, symbol='cross-thin', line=dict(width=1, color='black')),
        name='Detected Spikes',
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=thr_crossing_spikes_times[0],
        y=0*np.ones(len(thr_crossing_spikes_times[0])),
        mode='markers',
        marker=dict(color='magenta', size=7, symbol='circle', line=dict(width=1, color='black')),
        name='Thr Spikes',
        legendgroup="Thr_Spike",
        showlegend=True,
    ))
    for i in range(1, n_channels):
        fig.add_trace(go.Scatter(
            x=thr_crossing_spikes_times[i],
            y=i*np.ones(len(thr_crossing_spikes_times[i])),
            mode='markers',
            marker=dict(color='magenta', size=7, symbol='circle', line=dict(width=1, color='black')),
            name='Thr Spikes',
            legendgroup="Thr_Spike",
            showlegend=False,
        ))

    fig.update_layout(
        title="Spikes Overlaid on Pre-processed Data (Calculated via Templates)",
        xaxis_title="Samples",
        yaxis_title="Channel Index",
        template="plotly_white"
    )

    fig.show()

    breakpoint()


kilosort3_dirname = "/ceph/sjones/projects/sequence_squad/data/kilosort_output/EJT178/Recording6/2022-03-29_13-55-06/kilosort3"
plot_spikes_on_heatmap(kilosort3_dirname, sample_start=60000, n_samples=3000)

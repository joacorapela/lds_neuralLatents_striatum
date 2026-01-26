import sys
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def get_fig_spikes_on_heatmap(data_path, sample_start, n_samples,
                              thr_spikes_times, n_channels=383,
                              bytes_per_val=2):
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
    # sphike_templates.shape = n_spikes
    spike_templates = np.load(path / 'spike_templates.npy').flatten()

    # --- 4. Filter for our Time Window ---
    sample_end = sample_start + n_samples
    mask = (spike_times >= sample_start) & (spike_times < sample_end)
    
    window_spike_times = spike_times[mask]
    # window_spike_times = spike_times[mask] - sample_start
    # Get the peak channel for the template associated with each spike
    window_spike_chans = template_peak_chans[spike_templates[mask]]

    thr_spikes_times_flattened_lst = []
    thr_spikes_chans_lst = []
    for i in range(len(thr_spikes_times)):
        thr_spikes_times_flattened_lst.extend(thr_spikes_times[i])
        if i < 191: # channel 191 is the reference
            thr_spikes_chans_lst.extend((i*np.ones(len(thr_spikes_times[i]))).tolist())
        else:
            thr_spikes_chans_lst.extend(((i-1)*np.ones(len(thr_spikes_times[i]))).tolist())
    thr_spikes_times_flattened = np.array(thr_spikes_times_flattened_lst)
    thr_spikes_chans = np.array(thr_spikes_chans_lst)
    thr_mask = ((thr_spikes_times_flattened >= sample_start) &
                (thr_spikes_times_flattened < sample_end))
    window_thr_spikes_times = thr_spikes_times_flattened[thr_mask]
    window_thr_spikes_chans = thr_spikes_chans[thr_mask]

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

    # The KS Spike Markers
    fig.add_trace(go.Scatter(
        x=window_spike_times,
        y=window_spike_chans,
        mode='markers',
        marker=dict(color='yellow', size=12, symbol='circle', line=dict(width=1, color='black')),
        name='KS Spikes',
        showlegend=True,
    ))

    # The Thr Crossing Markers
    fig.add_trace(go.Scatter(
        x=window_thr_spikes_times,
        y=window_thr_spikes_chans ,
        mode='markers',
        marker=dict(color='magenta', size=7, symbol='circle', line=dict(width=1, color='black')),
        name='Thr Spikes',
        legendgroup="Thr_Spike",
        showlegend=True,
    ))

    fig.update_layout(
        title="Spikes Overlaid on Pre-processed Data",
        xaxis_title="Samples",
        yaxis_title="Channel Index",
        template="plotly_white"
    )

    return fig


def main(argv):
    thr_sample_start = 60000
    thr_n_samples = 30000
    thr_filename_pattern = "../../results/thrCrossingSpikes_startSample{:d}_endSample{:d}.pickle"
    kilosort3_dirname = "/ceph/sjones/projects/sequence_squad/data/kilosort_output/EJT178/Recording6/2022-03-29_13-55-06/kilosort3"
    fig_filename_pattern = "../../figures/ksAndThrSpikesOnHeatmap_startsample{:d}_endSample{:d}.{{:s}}"

    results_filename = thr_filename_pattern.format(
        thr_sample_start, thr_sample_start+thr_n_samples-1)
    fig_filename_pattern = fig_filename_pattern.format(thr_sample_start, thr_sample_start+thr_n_samples-1)

    with open(results_filename, "rb") as f:
        load_res = pickle.load(f)
    thr_spikes_times = load_res["spikes_times"]

    fig = get_fig_spikes_on_heatmap(
        data_path=kilosort3_dirname,
        sample_start=thr_sample_start,
        n_samples=thr_n_samples,
        thr_spikes_times=thr_spikes_times)
    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

import sys
import os.path
import importlib
import argparse
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


def get_fig_spikes_on_heatmap(kilosort3_dirname, thr_spikes_times,
                              start_time_sec, duration_sec,
                              bytes_per_val=2):
    path = Path(kilosort3_dirname)

    # get fs and n_channels from params.py
    params_filename = os.path.join(kilosort3_dirname, 'params.py')
    spec = importlib.util.spec_from_file_location("kilosort_params",
                                                  params_filename)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    fs = params.sample_rate
    n_channels = params.n_channels_dat

    sample_start = int(start_time_sec * fs)
    n_samples = int(duration_sec * fs)

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
    ks_spike_times = np.load(path / 'spike_times.npy').flatten()
    # Use spike_templates to get the original template index for each spike
    # sphike_templates.shape = n_spikes
    spike_templates = np.load(path / 'spike_templates.npy').flatten()

    # --- 4. Filter for our Time Window ---
    # first filter ks spikes
    sample_end = sample_start + n_samples
    ks_mask = (ks_spike_times >= sample_start) & (ks_spike_times < sample_end)

    ks_window_spike_times = ks_spike_times[ks_mask]

    # Get the peak channel for the template associated with each spike
    ks_window_spike_chans = template_peak_chans[spike_templates[ks_mask]]

    # now filter thr spikes
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
    thr_window_spikes_times = thr_spikes_times_flattened[thr_mask]
    thr_window_spikes_chans = thr_spikes_chans[thr_mask]

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
        x=ks_window_spike_times,
        y=ks_window_spike_chans,
        mode='markers',
        marker=dict(color='yellow', size=12, symbol='circle', line=dict(width=1, color='black')),
        name='KS Spikes',
        showlegend=True,
    ))

    # The Thr Crossing Markers
    fig.add_trace(go.Scatter(
        x=thr_window_spikes_times,
        y=thr_window_spikes_chans ,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr_start_time_sec", help="thresholded spikes start time (sec)",
                        type=float, default=5512.0)
    parser.add_argument("--thr_duration_sec", help="thresholded spikes duration (sec)",
                        type=float, default=120.0)
    parser.add_argument("--plot_start_time_sec", help="plot start time (sec)",
                        type=float, default=5512.0)
    parser.add_argument("--plot_duration_sec", help="plot duration (sec)",
                        type=float, default=5.0)
    parser.add_argument("--thr_filename_pattern",
                        help="threshold spikes filename", type=str,
                        default="../../results/EJT178_implant1/recording1_15-03-2022/thrCrossingSpikes_startTime{:.02f}_duration{:.02f}.pickle")
    parser.add_argument("--kilosort3_dirname", help="kilosort3 directory name",
                        type=str,
                        default="/ceph/sjones/projects/sequence_squad/data/kilosort_output/EJT178/Recording1/2022-03-15_13-55-36/kilosort3/")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/EJT178_implant1/recording1_15-03-2022/ksAndThrSpikesOnHeatmap_startsample{:.02f}_endSample{:.02f}.{{:s}}")
    args = parser.parse_args()

    thr_start_time_sec = args.thr_start_time_sec
    thr_duration_sec = args.thr_duration_sec
    plot_start_time_sec = args.plot_start_time_sec
    plot_duration_sec = args.plot_duration_sec
    thr_filename_pattern = args.thr_filename_pattern
    kilosort3_dirname = args.kilosort3_dirname
    fig_filename_pattern = args.fig_filename_pattern

    thr_results_filename = thr_filename_pattern.format(
        thr_start_time_sec, thr_duration_sec)
    fig_filename_pattern = fig_filename_pattern.format(plot_start_time_sec,
                                                       plot_duration_sec)

    with open(thr_results_filename, "rb") as f:
        thr_load_res = pickle.load(f)
    thr_spikes_times = thr_load_res["spikes_times"]

    fig = get_fig_spikes_on_heatmap(
        kilosort3_dirname=kilosort3_dirname,
        thr_spikes_times=thr_spikes_times,
        start_time_sec=plot_start_time_sec,
        duration_sec=plot_duration_sec)

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

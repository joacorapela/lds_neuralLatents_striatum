import sys
import os.path
import argparse
import importlib
import numpy as np
import plotly.graph_objects as go
import open_ephys.analysis


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_electrode", help="electrode where to find spikes",
                        type=int, default=70)
    parser.add_argument("--plot_start_time_sec", help="start time to plot (sec)",
                        type=float, default=5584.0)
    parser.add_argument("--plot_duration_sec", help="duration to plot (sec)",
                        type=float, default=2.0)
                        # type=float, default=120.0)
    parser.add_argument("--raw_ephys_data_dirname", type=str,
                        help="dirname containing the raw data",
                        default="/ceph/sjones/projects/sequence_squad/data/raw_neuropixel/OE_DATA/EJT178/290322/2022-03-29_13-55-06")
    parser.add_argument("--kilosort3_dirname", help="kilosort3 directory name",
                        type=str,
                        default="/ceph/sjones/projects/sequence_squad/data/kilosort_output/EJT178/Recording6/2022-03-29_13-55-06/kilosort3")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/spikesFromKilosort_startTime{:.02f}_endTime{:.02f}_channel{:03d}.{{:s}}")
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../results/kilosort3SpikesForElectrode_electrode{:03d}.npz")

    args = parser.parse_args()
    target_electrode = args.target_electrode
    plot_start_time_sec = args.plot_start_time_sec
    plot_end_time_sec = args.plot_start_time_sec + args.plot_duration_sec
    raw_ephys_data_dirname = args.raw_ephys_data_dirname
    kilosort3_dirname = args.kilosort3_dirname
    fig_filename_pattern = args.fig_filename_pattern.format(
        plot_start_time_sec, plot_end_time_sec, target_electrode)
    results_filename = args.results_filename_pattern.format(target_electrode)

    # get ap_band_start_time
    session_data = open_ephys.analysis.Session(raw_ephys_data_dirname)
    recording = session_data.recordnodes[0].recordings[0]
    recording.add_sync_line(1,102,"0",main=True)
    recording.add_sync_line(1,109,"0",main=False)
    recording.compute_global_timestamps()
    ap_stream = recording.continuous[0]
    ap_band_start_time = ap_stream.global_timestamps[0]

    # 1. Load the essential files
    spikes_times = np.load(os.path.join(kilosort3_dirname,
                                       'spike_times.npy')).flatten() # 33570299 spikes
    templates = np.load(os.path.join(kilosort3_dirname,
                                     'templates.npy')) # shape \in (nTemplates, nTimepoints, nTempChannels) = (982, 82, 383)
    templates_ind = np.load(os.path.join(kilosort3_dirname,
                                         'templates_ind.npy')).astype(int) # Maps template channels to actual channel IDs \in (982, 383)
    spike_templates = np.load(os.path.join(kilosort3_dirname, 'spike_templates.npy')).flatten().astype(int)
    winv = np.load(os.path.join(kilosort3_dirname,
                                'whitening_mat_inv.npy')) # Inverse whitening matrix

    # 2. Un-whiten the templates to get the real waveforms
    # Kilosort calculates templates in a whitened space. 
    # We reconstruct them: T_raw = T_whitened @ Winv
    n_templates, n_samples, n_temp_channels = templates.shape
    templates_unwhitened = np.zeros(templates.shape)

    for t in range(n_templates):
        # We only use the subset of the whitening matrix relevant to this template's channels
        chan_indices = templates_ind[t, :]
        templates_unwhitened[t, :, :] = np.dot(templates[t, :, :], winv[chan_indices, :][:, chan_indices])

    # 3. Find the peak channel for every template
    # Calculate peak-to-trough amplitude for each channel within the template
    amplitudes = np.max(templates_unwhitened, axis=1) - np.min(templates_unwhitened, axis=1) # shape = (982, 383)

    # Find which local channel index has the max amplitude
    best_chan_local_idx = np.argmax(amplitudes, axis=1)

    # Map that local index back to the actual electrode ID using templates_ind
    template_peak_channels = np.array([templates_ind[t, best_chan_local_idx[t]] for t in range(n_templates)])

    # 4. Filter spikes by your target electrode
    target_templates = np.where(template_peak_channels == target_electrode)[0]

    # 5. Extract the timestamps
    params_filename = os.path.join(kilosort3_dirname, 'params.py')
    spec = importlib.util.spec_from_file_location("kilosort_params",
                                                  params_filename)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    mask = np.isin(spike_templates, target_templates)
    target_spikes_times = (spikes_times[mask] / params.sample_rate) + ap_band_start_time # spikes_times is in samples units; converting to seconds

    print(f"Found {len(target_spikes_times)} spikes belonging to electrode {target_electrode}")

    np.save(file=results_filename, arr=target_spikes_times)
    print(f"Spike times saved to {results_filename}")

    spikes_times_to_plot = target_spikes_times[
        np.logical_and(plot_start_time_sec<=target_spikes_times,
                       target_spikes_times<=plot_end_time_sec)]
    # plotting
    fig = go.Figure()
    trace = go.Scatter(x=spikes_times_to_plot,
                       y=np.ones(len(spikes_times_to_plot)),
                       mode="markers")
    fig.add_trace(trace)
    for spike_time in spikes_times_to_plot:
        fig.add_vline(x=spike_time)
    fig.update_xaxes(title="Time (sec)")
    fig.update_layout(title=f"Electrode {target_electrode:03d}")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)

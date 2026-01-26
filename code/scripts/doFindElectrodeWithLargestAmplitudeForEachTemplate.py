
import sys
import os.path
import argparse
import importlib
import numpy as np
import pandas as pd
import open_ephys.analysis
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates_indices_filename",
                        help="filename with templates indices", type=str,
                        default="../../metadata/clustersIndices_124_223.ini")
    parser.add_argument("--kilosort3_dirname", help="kilosort3 directory name",
                        type=str,
                        default="/ceph/sjones/projects/sequence_squad/data/kilosort_output/EJT178/Recording6/2022-03-29_13-55-06/kilosort3")
    parser.add_argument("--results_filename", type=str,
                        help="results filename",
                        default="../../results/electrodeWithLargestAmplitudesForClusterIndices_124_233.ini")

    args = parser.parse_args()
    templates_indices_filename = args.templates_indices_filename
    kilosort3_dirname = args.kilosort3_dirname
    results_filename = args.results_filename

    templates_indices_df = pd.read_csv(templates_indices_filename, header=None)
    templates_indices = templates_indices_df.iloc[0].to_list()

    # 1. Load the essential files
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

    channels_ids = []
    for template_index in templates_indices:
        channels_ids.append(template_peak_channels[template_index].item())
    unique_channels_ids = list(set(channels_ids))

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)

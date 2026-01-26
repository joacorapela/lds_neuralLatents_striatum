import pickle
import numpy as np
from scipy.signal import butter, sosfiltfilt
from open_ephys.analysis import Session

sample_start = 60000
n_samples = 30000
results_filename_pattern = "../../results/thrCrossingSpikes_startSample{:d}_endSample{:d}.pickle"
results_filename = results_filename_pattern.format(sample_start, sample_start+n_samples-1)

# 1. Setup paths and load session
directory = "/ceph/sjones/projects/sequence_squad/data/raw_neuropixel/OE_DATA/EJT178/290322/2022-03-29_13-55-06"
session = Session(directory)
recording = session.recordnodes[0].recordings[0]
continuous = recording.continuous[0]

# Metadata
fs = continuous.metadata['sample_rate']
n_channels = continuous.metadata['num_channels']
total_samples = continuous.samples.shape[0]

# Subset of channels to estimate the median (e.g., every 20th channel)
ref_chans = np.linspace(0, n_channels - 1, 20).astype(int)

# 2. Parameters
chunk_samples = n_samples
threshold_multiplier = 5 
refractory_samples = int(0.0015 * fs) # 1.5ms

# 3. High-pass filter (300 Hz)
sos = butter(3, 300, 'hp', fs=fs, output='sos')

# 4. Initialize output: list of lists
spikes_times = [[] for _ in range(n_channels)]

print(f"Starting chunked detection. Total duration: {n_samples/fs:.2f}s")

# 5. Main Loop: Iterate through Time Chunks
for start in range(sample_start, sample_start+n_samples, chunk_samples):
    end = start + chunk_samples
    
    # A. Load reference subset for this chunk to calculate Median
    # Shape will be (n_samples, 20)
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

    progress = (end / (sample_start + n_samples)) * 100
    print(f"Progress: {progress:.1f}% | Processed up to {(end-sample_start)/fs:.1f}s")

print("Spike detection complete.")

results = {"spikes_times": spikes_times}
with open(results_filename, "wb") as f:
    pickle.dump(results, f)
print(f"Saved spike times to {results_filename}")
breakpoint()

import numpy as np
from scipy.signal import butter, sosfiltfilt
from open_ephys.analysis import Session

# 1. Setup paths and load session
directory = "/ceph/sjones/projects/sequence_squad/data/raw_neuropixel/OE_DATA/EJT178/290322/2022-03-29_13-55-06"
spikes_filename = "../../results/zero_crossing_spikes.npz"
session = Session(directory)
recording = session.recordnodes[0].recordings[0]
continuous = recording.continuous[0]

fs = continuous.metadata['sample_rate']
n_channels = continuous.metadata['num_channels']
total_samples = continuous.samples.shape[0]

# 1. Calculate the 'Global Reference' (Common Median)
# We take a subset of channels (e.g., 20 evenly spaced) to estimate the global noise
# This is much faster than loading all 384 channels at once.
print("Estimating global reference...")
sample_channels = np.linspace(0, n_channels - 1, 20).astype(int)
sample_data = continuous.get_samples(0, total_samples, selected_channels=sample_channels)
common_median = np.median(sample_data, axis=1) # The 'noise' shared by everyone

# 2. Define Parameters
threshold_multiplier = 5  # -5 * sigma
refractory_period_ms = 1.5
refractory_samples = int((refractory_period_ms / 1000) * fs)

# 3. High-pass filter (300 Hz) to remove LFP/Drift
# This is essential for clean thresholding
sos = butter(3, 300, 'hp', fs=fs, output='sos')

# 4. Initialize the output list
# spike_times[c] will be a list of sample indices for channel c
spike_times = [[] for _ in range(n_channels)]

print(f"Starting spike detection on {n_channels} channels...")

# 5. Main Loop: Process channel by channel
for c in range(n_channels):
    # Load only one channel at a time
    raw_data = continuous.get_samples(0, total_samples, selected_channels=[c]).flatten()
    
    # APPLY REFERENCE: Subtract the global noise
    referenced_data = raw_data - common_median# Apply filter

    # Filter the referenced signal
    filtered_data = sosfiltfilt(sos, referenced_data)

    # Calculate robust threshold (Median Absolute Deviation)
    sigma_n = np.median(np.abs(filtered_data)) / 0.6745
    thr = -5 * sigma_n

    crossings = np.where(filtered_data < thr)[0]
    
    # Apply refractory period
    if len(crossings) > 0:
        current_spike_list = [crossings[0]]
        for sample in crossings[1:]:
            if sample - current_spike_list[-1] > refractory_samples:
                current_spike_list.append(sample)
        
        spike_times[c] = current_spike_list

    if c % 10 == 0: # Progress update every 10 channels
        print(f"Processed channel {c}/{n_channels}...")

print("Detection complete.")

results = {"spikes_times": spikes_times}
np.savez(results_filename, results)
print(f"Saved spike times to {results_filename}")

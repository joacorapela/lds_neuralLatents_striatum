import numpy as np


def detect_mua_peaks(data, threshold, dead_time_ms, fs):
    # Convert dead time to samples (usually ~0.3ms for MUA)
    dead_time_samples = int(dead_time_ms * fs / 1000)

    # 1. Find all points below threshold
    below_thresh = data < threshold

    # 2. Find local minima: data[i] < data[i-1] and data[i] < data[i+1]
    # We use slices to do this very fast in NumPy
    is_minima = np.zeros_like(below_thresh)
    is_minima[1:-1] = (data[1:-1] < data[:-2]) & (data[1:-1] <= data[2:])

    # Candidates are local minima that are also below threshold
    spike_candidates = np.where(below_thresh & is_minima)[0]

    # 3. Minimal Lockout (to prevent noise-induced double peaks)
    final_spikes = []
    if len(spike_candidates) > 0:
        last_spike = -dead_time_samples
        for s in spike_candidates:
            if s - last_spike >= dead_time_samples:
                final_spikes.append(s)
                last_spike = s

    return np.array(final_spikes)




def epochLatents(latents, trials_times, bins_centers):
    # latents \in n_latents x T
    # bins_centers \in T
    # trials_times[r] \in trial_n_samples

    n_trials = len(trials_times)
    answer = [None] * n_trials
    for r in range(n_trials):
        answer[r] = latents[:, np.logical_and(
            trials_times[r][0] <= bins_centers,
            bins_centers < trials_times[r][-1])]
    return answer

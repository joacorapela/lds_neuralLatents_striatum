
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/spikes_startTime{:.02f}_endTime{:.02f}_channel{:03d}.{{:s}}")
    # plotting
    fig = go.Figure()
    trace = go.Scatter(x=ap_band_timestamps, y=ap_band_samples_muV,
                       mode="markers+lines")
    fig.add_trace(trace)
    fig.add_hline(y=threshold)
    for spike_index in spike_indices:
        fig.add_vline(x=ap_band_timestamps[spike_index])
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title=r"$\text{Voltage}\;(\mu V)$")
    fig.update_layout(title=f"Channel {channel:03d}")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()


import sys
import argparse
import plotly.graph_objects as go
import open_ephys.analysis


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_channels", help="selected channels to plot",
                        type=str, default="[10,110,210]")
    parser.add_argument("--start_time_sec", help="start time to plot (sec)",
                        type=float, default=5512.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        type=float, default=5.0)
                        # type=float, default=120.0)
    parser.add_argument("--raw_ephys_data_folder", type=str,
                        help="folder containing the raw data",
                        default="/ceph/sjones/projects/sequence_squad/data/raw_neuropixel/OE_DATA/EJT178/290322/2022-03-29_13-55-06")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/EJT178_2022-03-29_13-55-06_startSec{:.02f}_endSec{:02f}_channels{:s}_means.{{:s}}")
    args = parser.parse_args()

    selected_channels = [int(selected_channel)
                         for selected_channel
                         in args.selected_channels[1:-1].split(",")]
    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    raw_ephys_data_folder = args.raw_ephys_data_folder
    fig_filename_pattern = args.fig_filename_pattern.format(start_time_sec,
                                                            end_time_sec,
                                                            args.selected_channels[1:-1])

    print(f"Loading and processing ephys data at {raw_ephys_data_folder}")
    session_data = open_ephys.analysis.Session(raw_ephys_data_folder)

    # len(session_data.recordnodes): 1
    # len(session_data.recordnodes[0].recordings): 1
    recording = session_data.recordnodes[0].recordings[0]
    recording.add_sync_line(1,102,"0",main=True)
    recording.add_sync_line(1,109,"1",main=False)
    recording.add_sync_line(1,109,"0",main=False)

    recording.compute_global_timestamps()
    # len(session_data.recordnodes[0].recordings[0].continuous): 3
    #
    # session_data.recordnodes[0].recordings[0].continuous[0]: AP stream from probe Neuropix-PXI-109
    # print(session_data.recordnodes[0].recordings[0].continuous[0].metadata["source_node_name"])
    # Neuropix-PXI
    # print(session_data.recordnodes[0].recordings[0].continuous[0].metadata["sample_rate"])
    # 30000
    #
    # session_data.recordnodes[0].recordings[0].continuous[1]: LFP stream from probe Neuropix-PXI-109
    # print(session_data.recordnodes[0].recordings[0].continuous[0].metadata["source_node_name"])
    # Neuropix-PXI
    # print(session_data.recordnodes[0].recordings[0].continuous[1].metadata["sample_rate"])
    # 2500
    #
    # session_data.recordnodes[0].recordings[0].continuous[2]: NI-DAQmx

    # Define the streams
    ap_stream = recording.continuous[0]

    ap_band_sample_rate = ap_stream.metadata["sample_rate"]
    ap_band_start_time = ap_stream.global_timestamps[0]
    if start_time_sec < ap_band_start_time:
        raise ValueError(f"start_time_sec={start_time_sec} should be >= than "
                         f"ap_band_start_time={ap_band_start_time}")
    ap_band_end_time = ap_stream.global_timestamps[-1]
    if end_time_sec > ap_band_end_time:
        raise ValueError(f"end_time_sec={end_time_sec} should be <= than "
                         f"ap_band_end_time={ap_band_end_time}")
    ap_band_start_sample = int((start_time_sec-ap_band_start_time) * ap_band_sample_rate)
    ap_band_end_sample = int((end_time_sec-ap_band_start_time) * ap_band_sample_rate)
    ap_band_samples_muV = ap_stream.get_samples(
        start_sample_index=ap_band_start_sample,
        end_sample_index=ap_band_end_sample,
        selected_channels=selected_channels,
    )
    ap_band_means = ap_band_samples_muV.mean(axis=0)

    fig = go.Figure()
    trace = go.Bar(x=args.selected_channels[1:-1].split(","), y=ap_band_means)
    fig.add_trace(trace)
    fig.update_xaxes(title="Channel")
    fig.update_yaxes(title=r"$\text{Mean Voltage}\;\mu V")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)


import sys
import argparse
import numpy as np
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtering_res_number", type=int,
                        help="number corresponding to filtered results filename",
                        default=42904130)
    parser.add_argument("--start_time_sec", help="start time to plot (sec)",
                        type=float, default=4045.0)
    parser.add_argument("--duration_sec", help="duration to plot (sec)",
                        type=float, default=300.0)
    parser.add_argument("--hmm_params_filename_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_params_from{:.02f}_to{:.02f}.npz")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_from{:.02f}_to{:.02f}_{{:s}}.{{:s}}")

    args = parser.parse_args()

    filtering_res_number = args.filtering_res_number
    start_time_sec = args.start_time_sec
    end_time_sec = args.start_time_sec + args.duration_sec
    hmm_params_filename = args.hmm_params_filename_pattern.format(
        filtering_res_number, start_time_sec, end_time_sec)
    fig_filename_pattern = args.fig_filename_pattern.format(
        filtering_res_number, start_time_sec, end_time_sec)

    load_res = np.load(hmm_params_filename)
    A = load_res["A"] # \in (K, K)
    means = load_res["means"] # \in (K, R)
    covs = load_res["covs"] # \in (K, K, R)
    R = means.shape[1]

    latents = np.arange(R)
    stds = np.sqrt(np.diagonal(covs, axis1=0, axis2=1)).T

    fig = go.Figure()
    trace = go.Heatmap(z=A)
    fig.add_trace(trace)
    fig.update_xaxes(title="To State")
    fig.update_yaxes(title="From State")
    fig.write_image(fig_filename_pattern.format("transition", "png"))
    fig.write_html(fig_filename_pattern.format("transition", "html"))
    print(f'saved {fig_filename_pattern.format("transition", "html")}')
    fig.show()

    fig = go.Figure()
    for i in range(R):
        trace = go.Bar(x=latents, y=means[:, i], name=f"State {i}",
                       error_y=dict(type='data', array=stds[:, i]))
        fig.add_trace(trace)
    fig.update_xaxes(title="Latent Index")
    fig.update_yaxes(title=r"$\text{Mean}\pm\text{STD}$")
    fig.write_image(fig_filename_pattern.format("mean", "png"))
    fig.write_html(fig_filename_pattern.format("mean", "html"))
    print(f'saved {fig_filename_pattern.format("mean", "html")}')
    fig.show()

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)

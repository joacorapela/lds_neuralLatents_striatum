
import sys
import pickle
import argparse
import numpy as np
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="model estimation number",
                        type=int, default=82227365)
    parser.add_argument("--estimation_results_filename_pattern", type=str,
                        default="../../results/{:08d}_estimation.{:s}")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/{:08d}_estimation_logLike_vs_{:s}.{:s}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    estimation_results_filename_pattern = args.estimation_results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    estimation_results_filename = estimation_results_filename_pattern.format(
        est_res_number, "pickle")
    iter_html_fig_filename = fig_filename_pattern.format(est_res_number, "iter", "html")
    iter_png_fig_filename = fig_filename_pattern.format(est_res_number, "iter", "png")
    elapsed_html_fig_filename = fig_filename_pattern.format(est_res_number, "elapsed", "html")
    elapsed_png_fig_filename = fig_filename_pattern.format(est_res_number, "elapsed", "png")

    with open(estimation_results_filename, "rb") as f:
        est_results = pickle.load(f)

    log_like = est_results["log_like"]
    elapsed_time = est_results["elapsed_time"]
    iter_numbers = np.arange(len(log_like))

    fig = go.Figure()
    trace = go.Scatter(x=iter_numbers, y=log_like, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title="Iteration Number")
    fig.update_yaxes(title="Log Likelihood")

    fig.write_html(iter_html_fig_filename)
    fig.write_image(iter_png_fig_filename)

    fig = go.Figure()
    trace = go.Scatter(x=elapsed_time, y=log_like, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title="Elapsed Time")
    fig.update_yaxes(title="Log Likelihood")

    fig.write_html(elapsed_html_fig_filename)
    fig.write_image(elapsed_png_fig_filename)

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)

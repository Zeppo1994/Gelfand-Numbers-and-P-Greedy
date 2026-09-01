"""Legendre P-greedy sampling curves and covariance-tail lower bounds."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import LegendreMercerKernel
import lower_bounds as bounds

SMOOTHNESS_VALUES = (2.0, 3.0)
CANDIDATE_GRID_SIZE = 20_000
POINT_DESIGN_SIZE = 64


def _comparison_panel(
    ax,
    *,
    s=2.0,
    max_iter=1000,
    sel_grid=CANDIDATE_GRID_SIZE,
    n_cap=1000,
):
    kernel = LegendreMercerKernel(s=s)
    result = bounds.sampling_vs_lower_bound(
        kernel,
        m_max=max_iter,
        sel_grid=sel_grid,
        n_cap=n_cap,
    )
    lower_n = result["lower_n"]
    lower = result["lower"]

    bounds.plot_sampling_estimate(ax, result["sampling_n"], result["sampling"])
    bounds.plot_gelfand_lower_bound(ax, lower_n, lower)

    anchor = min(3, len(lower_n) - 1)
    rate_indices = np.arange(2, result["n_used"])
    ax.loglog(
        rate_indices,
        lower[anchor] * (rate_indices / lower_n[anchor]) ** (-(s - 0.5)),
        "k--",
        lw=1.0,
        label=rf"$\propto n^{{-{s - 0.5:g}}}$",
    )
    ax.set_xlabel(r"$m$ (points) $/$ $n$ (width)")
    ax.set_ylabel(r"$\|\cdot\|_\infty$ width")
    ax.set_title(rf"Legendre RKHS, $s={s:g}$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return result


def comparison_figure(
    smoothness_values=SMOOTHNESS_VALUES,
    max_iter=1000,
    sel_grid=CANDIDATE_GRID_SIZE,
    n_cap=1000,
    out="figures/legendre.png",
):
    """Generate the two-panel lower-bound comparison."""
    smoothness_values = tuple(float(s) for s in smoothness_values)
    if not smoothness_values:
        raise ValueError("smoothness_values must not be empty")

    fig, axes = plt.subplots(
        1,
        len(smoothness_values),
        figsize=(5.0 * len(smoothness_values), 4.2),
        squeeze=False,
    )
    combined = {}
    for axis, smoothness in zip(axes[0], smoothness_values):
        result = _comparison_panel(
            axis,
            s=smoothness,
            max_iter=max_iter,
            sel_grid=sel_grid,
            n_cap=n_cap,
        )
        tag = f"s{smoothness:g}".replace(".", "p")
        bounds.add_prefixed_result(combined, tag, result)
        print(
            f"panel {tag}: n_used={result['n_used']}, sampling grid={sel_grid}, "
            "median P-greedy/lower-bound ratio="
            f"{np.nanmedian(result['ratio']):.2f}"
        )

    output = bounds.finalize_figure(fig, out)
    print(f"figure saved -> {output}")
    return combined


def points_figure(
    smoothness_values=SMOOTHNESS_VALUES,
    m=POINT_DESIGN_SIZE,
    grid=CANDIDATE_GRID_SIZE,
    out="figures/legendre_points.png",
):
    """Plot Legendre P-greedy selection order for each smoothness."""
    smoothness_values = tuple(float(s) for s in smoothness_values)
    m = int(m)
    if not smoothness_values:
        raise ValueError("smoothness_values must not be empty")
    if m < 1:
        raise ValueError("m must be positive")

    max_iter = int(np.ceil(1.2 * m))
    fig, axes = plt.subplots(
        1,
        len(smoothness_values),
        figsize=(5.0 * len(smoothness_values), 4.8),
        squeeze=False,
    )
    designs = {}
    for axis, smoothness in zip(axes[0], smoothness_values):
        kernel = LegendreMercerKernel(s=smoothness)
        greedy = bounds.fit_p_greedy(
            kernel,
            max_iter=max_iter,
            sel_grid=grid,
        )
        if greedy.n_ < m:
            plt.close(fig)
            raise RuntimeError(
                f"P-greedy stopped at {greedy.n_} centers before m={m} "
                f"for s={smoothness:g}"
            )

        centers = greedy.ctrs_.reshape(-1).cpu().numpy()
        design = centers[:m]
        order = np.arange(1, len(centers) + 1)
        axis.scatter(centers, order, c=order, cmap="viridis", s=16, zorder=2)
        axis.axhline(m, color="C3", lw=1.2, ls="--", zorder=3)
        axis.plot(
            design,
            np.full(m, m),
            "|",
            color="C3",
            ms=9,
            mew=1.4,
            label=rf"first $m={m}$ centers",
            zorder=4,
        )
        axis.set_xlabel(r"center location $x_i$")
        axis.set_ylabel("selection step")
        axis.set_title(rf"Legendre kernel: $s={smoothness:g}$, $m={m}$")
        axis.set_xlim(-1.03, 1.03)
        axis.set_ylim(0, 1.03 * len(centers))
        axis.legend(fontsize=8, loc="lower right")
        axis.grid(True, alpha=0.3)

        tag = f"s{smoothness:g}".replace(".", "p")
        designs[f"{tag}_centers"] = centers
        designs[f"{tag}_n_used"] = greedy.n_

    fig.suptitle("Endpoint clustering in Legendre P-greedy designs")
    output = bounds.finalize_figure(fig, out)
    print(f"figure saved -> {output}")
    return designs


if __name__ == "__main__":
    comparison_figure()
    points_figure()

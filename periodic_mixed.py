"""Periodic mixed-Sobolev P-greedy curves and Fourier-tail lower bounds."""

from __future__ import annotations

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import PeriodicSobolevMixedKernel
import lower_bounds as bounds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POWER_FLOOR = 2e-6
CASES = [
    (2, {1: 12_000, 2: 5_000, 3: 1_000}, 120_000, 1_000),
    (3, {1: 10_000, 2: 4_000}, 80_000, 1_000),
]


def _benchmark_rate(indices, smoothness, dimension):
    """Return n^{-(m-1/2)} (log n)^{(d-1)m}."""
    indices = np.asarray(indices, dtype=float)
    return indices ** (-(smoothness - 0.5)) * np.log(indices) ** (
        (dimension - 1) * smoothness
    )


def _fit_constant(indices, values, smoothness, dimension):
    """Fit one vertical offset to the asymptotic guide in log space."""
    return np.exp(
        np.mean(
            np.log(values) - np.log(_benchmark_rate(indices, smoothness, dimension))
        )
    )


def comparison_figure(
    *,
    out="figures/periodic_mixed.png",
):
    """Generate the lower-bound comparison in dimensions two and three."""
    colors = {1: "C0", 2: "C1", 3: "C2"}
    fig, axes = plt.subplots(
        1, len(CASES), figsize=(7.0 * len(CASES), 4.7), squeeze=False
    )
    combined = {}

    for axis, (d, node_counts, selection_grid, n_cap) in zip(axes[0], CASES):
        print(f"\n=== d={d} ===  (greedy grid {selection_grid}, dev={DEVICE})")
        for smoothness, requested_nodes in node_counts.items():
            dtype = torch.float32 if smoothness == 1 else torch.float64
            kernel = PeriodicSobolevMixedKernel(
                m=smoothness,
                d=d,
                dtype=dtype,
                device=DEVICE,
            )
            greedy = bounds.fit_p_greedy(
                kernel,
                d=d,
                max_iter=requested_nodes,
                sel_grid=selection_grid,
                device=DEVICE,
            )
            sampling = greedy.g_curve().cpu().double().numpy()
            n_used = greedy.n_
            indices = np.arange(len(sampling))
            reliable = np.flatnonzero(sampling > POWER_FLOOR)
            reliable_end = int(reliable[-1]) if reliable.size else n_used - 1
            fit_indices = bounds.log_spaced_ints(
                min(n_cap, n_used - 1, reliable_end), 14
            )

            lower_tails = kernel.gelfand_lower_tails(n_used)
            lower = np.sqrt(np.clip(lower_tails, 0.0, None))
            lower_at_fit = lower[fit_indices]
            peak = np.exp((d - 1) * smoothness / (smoothness - 0.5))
            start = max(2.0 * peak, fit_indices[0])
            fit_mask = (fit_indices >= start) & (lower_at_fit > 0.0)
            if fit_mask.sum() < 2:
                fit_mask = lower_at_fit > 0.0
            constant = _fit_constant(
                fit_indices[fit_mask],
                lower_at_fit[fit_mask],
                smoothness,
                d,
            )
            guide_indices = np.geomspace(max(start, 2.0), indices[-1], 60)
            positive = lower[1:] > 0.0
            color = colors[smoothness]

            axis.loglog(
                indices[2:],
                sampling[2:],
                "o-",
                markevery=max(1, len(indices[2:]) // 45),
                ms=3.0,
                color=color,
                lw=1.6,
                label=rf"$g_n^{{\mathrm{{lin}}}}$ (P-greedy), $m={smoothness}$",
            )
            lower_indices = np.arange(1, len(lower))[positive]
            bounds.plot_gelfand_lower_bound(
                axis,
                lower_indices,
                lower[1:][positive],
                color=color,
                label=(
                    f"lower bound {bounds.LOWER_BOUND_SYMBOL} ≤ cₙ, "
                    rf"$m={smoothness}$"
                ),
            )
            axis.loglog(
                guide_indices,
                constant * _benchmark_rate(guide_indices, smoothness, d),
                ":",
                color=color,
                lw=1.4,
                label=(
                    rf"$n^{{-{smoothness - 0.5:g}}}(\log n)^"
                    rf"{{{(d - 1) * smoothness}}}$, $m={smoothness}$"
                ),
            )

            sampling_at_fit = sampling[fit_indices]
            print(
                f"  m={smoothness}: n_used={n_used:4d}  "
                "median P-greedy/lower-bound ratio="
                f"{np.nanmedian(sampling_at_fit / lower_at_fit):.2f}"
            )
            tag = f"d{d}_m{smoothness}"
            combined[f"{tag}_sampling_n"] = indices
            combined[f"{tag}_sampling"] = sampling
            combined[f"{tag}_lower"] = lower
            combined[f"{tag}_n_used"] = n_used

        bounds.finish_comparison_axis(
            axis,
            rf"$H^m_{{\mathrm{{mix}}}}([0,1]^{d})$: "
            "sampling estimates and lower bounds",
            xlabel=r"$n$ (points / width index)",
            legend_fontsize=7.2,
        )

    output = bounds.finalize_figure(fig, out)
    print(f"figure saved -> {output}")
    return combined


def points_figure(
    *,
    smoothness=2,
    counts=(64, 256, 576),
    sel_grid=24_000,
    out="figures/periodic_mixed_points.png",
):
    """Plot nested two-dimensional periodic P-greedy designs."""
    counts = np.asarray(tuple(int(count) for count in counts), dtype=int)
    if counts.size == 0 or np.any(counts < 1):
        raise ValueError("counts must contain positive integers")

    kernel = PeriodicSobolevMixedKernel(
        m=int(smoothness),
        d=2,
        dtype=torch.float64,
        device=DEVICE,
    )
    greedy = bounds.fit_p_greedy(
        kernel,
        d=2,
        max_iter=int(counts.max()),
        sel_grid=sel_grid,
        device=DEVICE,
    )
    if greedy.n_ < counts.max():
        raise RuntimeError(
            f"P-greedy stopped at {greedy.n_} centers before n={counts.max()}"
        )

    centers = greedy.ctrs_.cpu().numpy()
    fig, axes = plt.subplots(
        1,
        len(counts),
        figsize=(4.6 * len(counts), 4.3),
        squeeze=False,
    )
    for axis, count in zip(axes[0], counts):
        design = centers[:count]
        axis.scatter(
            design[:, 0],
            design[:, 1],
            c=np.arange(1, count + 1),
            cmap="viridis",
            s=max(3.0, 24.0 * (64.0 / count) ** 0.35),
        )
        axis.set_xlabel(r"$x_1$")
        axis.set_ylabel(r"$x_2$")
        axis.set_title(rf"first $n={count}$ centers")
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.25)

    fig.suptitle(
        rf"Nested periodic P-greedy designs, $H^{{{int(smoothness)}}}_"
        r"{\mathrm{mix}}([0,1]^2)$"
    )
    output = bounds.finalize_figure(fig, out)
    print(f"figure saved -> {output}")
    return {
        "centers": centers,
        "counts": counts,
        "n_used": greedy.n_,
    }


if __name__ == "__main__":
    comparison_figure()
    points_figure()

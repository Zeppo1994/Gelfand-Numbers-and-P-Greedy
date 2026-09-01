"""Paley-Wiener P-greedy sampling curves and covariance-tail lower bounds."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import PaleyWienerSincKernel
import lower_bounds as bounds

SPECS = [
    (20, dict(m_max=70, sel_grid=8000, n_cap=34)),
    (40, dict(m_max=110, sel_grid=8000, n_cap=56)),
    (80, dict(m_max=140, sel_grid=10000, n_cap=95)),
]


def comparison_figure(
    *,
    out="figures/paley_wiener.png",
):
    """Generate the three-bandwidth lower-bound comparison."""
    results = []
    combined = {}
    for n_eff, config in SPECS:
        kernel = PaleyWienerSincKernel(n_eff=n_eff)
        result = bounds.sampling_vs_lower_bound(kernel, **config)
        result["n_eff"] = kernel.n_eff
        result["diag"] = kernel.diag_val
        threshold = (
            20.0 * np.sqrt(np.finfo(np.float64).eps) * np.sqrt(result["diag"])
        )
        reliable = (
            (result["sampling_at_lower_n"] > threshold)
            & (result["lower"] > threshold)
        )
        result["reliable"] = reliable
        result["threshold"] = threshold
        results.append((n_eff, result))
        bounds.add_prefixed_result(combined, f"ne{n_eff}", result)

        if reliable.any():
            print(
                f"N_eff={n_eff:3d}  reliable n<="
                f" {int(result['lower_n'][reliable].max()):3d}  "
                "median P-greedy/lower-bound ratio="
                f" {np.nanmedian(result['ratio'][reliable]):.2f}"
            )

    fig, axes = plt.subplots(
        1, len(SPECS), figsize=(5.4 * len(SPECS), 4.5), squeeze=False
    )
    for axis, (n_eff, result) in zip(axes[0], results):
        panel_floor = 0.3 * result["threshold"]
        sampling_n, sampling = bounds.clipped_prefix(
            result["sampling_n"], result["sampling"], panel_floor
        )
        lower_n, lower = bounds.clipped_prefix(
            result["lower_n"], result["lower"], panel_floor
        )
        bounds.plot_sampling_estimate(axis, sampling_n, sampling)
        bounds.plot_gelfand_lower_bound(axis, lower_n, lower)
        axis.axvline(n_eff, color="0.4", ls="--", lw=1.0)
        axis.text(
            n_eff * 1.02,
            result["threshold"] * 3,
            rf"$N_{{\mathrm{{eff}}}}={n_eff}$",
            rotation=90,
            va="bottom",
            fontsize=8,
            color="0.35",
        )
        axis.axhspan(
            panel_floor, result["threshold"], color="0.85", alpha=0.6, lw=0
        )
        axis.text(
            0.97,
            0.03,
            f"P-greedy float64 floor ~{result['threshold']:.0e}",
            transform=axis.transAxes,
            fontsize=7,
            color="0.4",
            ha="right",
            va="bottom",
        )
        axis.set_ylim(panel_floor, 8.0)
        axis.set_xlabel(r"$m$ (points) $/$ $n$ (width)")
        axis.set_ylabel(r"$\|\cdot\|_\infty$ width")
        c_value = n_eff * np.pi / 2
        axis.set_title(
            rf"Paley-Wiener $K_c$, $c={c_value:.1f}$"
            rf"  ($N_{{\mathrm{{eff}}}}={n_eff}$)"
        )
        axis.legend(fontsize=7.5, loc="lower left")
        axis.grid(True, which="both", alpha=0.3)

    output = bounds.finalize_figure(fig, out)
    print(f"figure saved -> {output}")
    return combined


def points_figure(
    *,
    n_effs=(20, 40, 80),
    grid=6_000,
    out="figures/paley_wiener_points.png",
):
    """Plot P-greedy selection order at the three effective bandwidths."""
    n_effs = tuple(int(n_eff) for n_eff in n_effs)
    if not n_effs or any(n_eff < 1 for n_eff in n_effs):
        raise ValueError("n_effs must contain positive integers")

    fig, axes = plt.subplots(
        1,
        len(n_effs),
        figsize=(5.0 * len(n_effs), 4.6),
        squeeze=False,
    )
    designs = {}
    for axis, n_eff in zip(axes[0], n_effs):
        target = n_eff
        kernel = PaleyWienerSincKernel(n_eff=n_eff)
        greedy = bounds.fit_p_greedy(
            kernel,
            max_iter=int(np.ceil(1.2 * target)),
            sel_grid=grid,
        )
        if greedy.n_ < target:
            plt.close(fig)
            raise RuntimeError(
                f"P-greedy stopped at {greedy.n_} centers before n={target}"
            )

        centers = greedy.ctrs_.reshape(-1).cpu().numpy()
        design = np.sort(centers[:target])
        ideal = -1.0 + (2.0 * np.arange(target) + 1.0) / target
        order = np.arange(1, len(centers) + 1)
        axis.scatter(centers, order, c=order, cmap="viridis", s=16, zorder=2)
        axis.axhline(target, color="C3", lw=1.2, ls="--", zorder=3)
        axis.plot(
            centers[:target],
            np.full(target, target),
            "|",
            color="C3",
            ms=9,
            mew=1.4,
            label=rf"first $N_{{\mathrm{{eff}}}}={target}$ centers",
            zorder=4,
        )
        axis.set_xlabel(r"center location $x_i$")
        axis.set_ylabel("selection step")
        axis.set_title(rf"$N_{{\mathrm{{eff}}}}={n_eff}$")
        axis.set_xlim(-1.03, 1.03)
        axis.set_ylim(0, 1.03 * len(centers))
        axis.legend(fontsize=8, loc="lower right")
        axis.grid(True, alpha=0.3)

        tag = f"ne{n_eff}"
        designs[f"{tag}_centers"] = centers
        designs[f"{tag}_target"] = target
        designs[f"{tag}_max_uniform_deviation"] = np.max(np.abs(design - ideal))
        designs[f"{tag}_n_used"] = greedy.n_

    fig.suptitle("Paley–Wiener P-greedy designs across the spectral cliff")
    output = bounds.finalize_figure(fig, out)
    print(f"figure saved -> {output}")
    return designs


if __name__ == "__main__":
    comparison_figure()
    points_figure()

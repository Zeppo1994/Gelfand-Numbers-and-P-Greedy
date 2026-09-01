"""Matérn P-greedy sampling curves and discrete covariance-tail lower bounds."""

from __future__ import annotations

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greedy import power_function
from kernels import MaternKernel
import lower_bounds as bounds

SPECS = [
    (
        1,
        MaternKernel(nu=1.5, ell=0.3),
        dict(m_max=460, sel_grid=16_000, n_cap=300),
        dict(d=1, n_quad=512),
    ),
    (
        3,
        MaternKernel(nu=1.5, ell=0.6),
        dict(m_max=400, sel_grid=20_000, n_cap=140),
        dict(d=3, n_quad=512),
    ),
]


def comparison_figure(
    *,
    out="figures/matern.png",
):
    """Generate the one- and three-dimensional lower-bound comparison."""
    results = []
    combined = {}
    for d, kernel, sampling_config, lower_config in SPECS:
        result = bounds.sampling_vs_lower_bound(
            kernel,
            d=d,
            lower_bound_kwargs=lower_config,
            **sampling_config,
        )
        results.append((d, result))
        bounds.add_prefixed_result(combined, f"d{d}", result)
        print(
            f"Matérn d={d}: n_used={result['n_used']:4d}, "
            "median P-greedy/lower-bound ratio="
            f"{np.nanmedian(result['ratio']):.2f}"
        )

    fig, axes = plt.subplots(
        1, len(SPECS), figsize=(5.0 * len(SPECS), 4.2), squeeze=False
    )
    for axis, (d, result) in zip(axes[0], results):
        bounds.plot_sampling_estimate(axis, result["sampling_n"], result["sampling"])
        bounds.plot_gelfand_lower_bound(axis, result["lower_n"], result["lower"])
        axis.set_xlabel(r"$m$ (points) $/$ $n$ (width)")
        axis.set_ylabel(r"$\|\cdot\|_\infty$ width")
        axis.set_title(rf"Matérn $\nu=3/2$, $d={d}$")
        axis.legend(fontsize=8)
        axis.grid(True, which="both", alpha=0.3)

    output = bounds.finalize_figure(fig, out)
    print(f"figure saved -> {output}")
    return combined


def points_figure(
    *,
    n_points=128,
    snapshot_size=12,
    grid=4_000,
    query_grid=4_000,
    out="figures/matern_points.png",
):
    """Plot a one-dimensional Matérn design and an intermediate power function."""
    n_points = int(n_points)
    snapshot_size = int(snapshot_size)
    if not 1 <= snapshot_size < n_points:
        raise ValueError("snapshot_size must satisfy 1 <= snapshot_size < n_points")

    kernel = MaternKernel(nu=1.5, ell=0.3)
    greedy = bounds.fit_p_greedy(
        kernel,
        max_iter=n_points,
        sel_grid=grid,
    )
    if greedy.n_ < n_points:
        raise RuntimeError(
            f"P-greedy stopped at {greedy.n_} centers before n_points={n_points}"
        )

    centers = greedy.ctrs_.reshape(-1)
    snapshot = centers[:snapshot_size]
    query = torch.linspace(
        *kernel.domain,
        int(query_grid),
        dtype=kernel.dtype,
        device=centers.device,
    )[:, None]
    power = power_function(kernel, snapshot[:, None], query)
    centers_np = centers.cpu().numpy()
    query_np = query[:, 0].cpu().numpy()
    power_np = power.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(query_np, power_np, color="C0", lw=1.6)
    axes[0].plot(
        snapshot.cpu().numpy(),
        np.zeros(snapshot_size),
        "|",
        color="C3",
        ms=10,
        mew=1.4,
        label=rf"first {snapshot_size} centers",
    )
    axes[0].axvline(centers_np[snapshot_size], color="C2", ls="--", lw=1.2)
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$P_{X_m}(x)$")
    axes[0].set_title(rf"Power function after $m={snapshot_size}$ selections")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    order = np.arange(1, n_points + 1)
    axes[1].scatter(centers_np, order, c=order, cmap="viridis", s=16)
    axes[1].set_xlabel(r"center location $x_i$")
    axes[1].set_ylabel("selection step")
    axes[1].set_title(rf"Matérn P-greedy design, $n={n_points}$")
    axes[1].set_xlim(-1.03, 1.03)
    axes[1].grid(True, alpha=0.3)

    output = bounds.finalize_figure(fig, out)
    print(f"figure saved -> {output}")
    return {
        "centers": centers_np,
        "query": query_np,
        "power": power_np,
        "snapshot_size": snapshot_size,
        "n_used": greedy.n_,
    }


if __name__ == "__main__":
    comparison_figure()
    points_figure()

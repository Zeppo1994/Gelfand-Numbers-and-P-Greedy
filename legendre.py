"""
Driver: assess the sampling-vs-Gelfand comparison from the companion manuscript on the
univariate Legendre Mercer RKHS (the kernel example is from arXiv:2103.11124, Section 7.3).

Plots, in the paper's notation:
  * g_m^lin       sampling numbers, numerically estimated by the grid-resolved power function of
                  the computed P-greedy design (an upper surrogate, not the exact infimum).
  * c_n           Gelfand widths d_n(K)_H, the numerical Kolmogorov width of the translates.
  * the manuscript rate guide n^-(s-1/2).

Result: the estimated sampling numbers and c_n run parallel on n^-(s-1/2), with no observed sqrt(n)
growth; this is numerical evidence for the comparison theorem, not an exact computation of g_m^lin.
Small late-index oscillations in the scaled sampling curve are finite-m effects; for s=3 the
displayed endpoint is additionally limited by the squared-power stopping tolerance.
The s=3 Gelfand upper estimate reaches the float64 feature-residual cancellation floor near
1e-6; its late plateau is numerical, while the rigorous lower value remains informative.

Running this module produces the two-panel comparison figure ``figures/legendre.png`` and
the joint two-panel design figure ``figures/legendre_points.png`` for s=2 and s=3.
Both figures use the 20,000-point Chebyshev grid and Mercer truncation M=24,000.  s=2 keeps Pow^2 above the
float64 cancellation floor out to large m; smoother kernels stop earlier as Pow^2 -> 0 hits tol_p.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import LegendreMercerKernel
from greedy import PGreedy
import widths

torch.manual_seed(0)

SMOOTHNESS_VALUES = (2.0, 3.0)
MERCER_TRUNCATION = 24_000
CANDIDATE_GRID_SIZE = 20_000
ENDPOINT_LADDER_SIZE = 480
POINT_DESIGN_SIZE = 64

# Publication run: M=24000 keeps the omitted endpoint-energy scale below the smallest
# widths tracked near n=1000.  This Mercer truncation is specific to the Legendre feature
# expansion; similarly sized values in other drivers are sampling-node budgets, not M.

def _rates_panel(
    ax,
    s=2.0,
    n_trunc=MERCER_TRUNCATION,
    max_iter=1000,
    sel_grid=CANDIDATE_GRID_SIZE,
    cn_grid=5000,
    n_cap=1000,
    edge_ladder=ENDPOINT_LADDER_SIZE,
    compress_irls=True,
):
    """Draw one Legendre sampling-vs-Gelfand panel on ``ax``."""
    kernel = LegendreMercerKernel(s=s, n_trunc=n_trunc)
    r = widths.sampling_vs_gelfand(
        kernel, m_max=max_iter, sel_grid=sel_grid, cn_grid=cn_grid, n_cap=n_cap,
        edge_ladder=edge_ladder,
        compress_irls=compress_irls,
    )
    m_list, g, cN, cn, n_used = r["m_list"], r["g"], r["cN"], r["cn"], r["n_used"]
    cn_lo = r["cn_lo"]  # c_n^- is rigorous; c_n^+ is certified only where flagged
    cn_mid = np.sqrt(
        cn_lo * cn
    )  # geometric-mean representative for the manuscript rate guide

    # --- plot: estimated sampling numbers vs Gelfand widths ---
    ax.loglog(
        m_list,
        g,
        "o-",
        ms=3.5,
        color="C0",
        label=r"$g_m^{\mathrm{lin}}$  (P-greedy estimate)",
    )
    # c_n^- is rigorous; the helper draws the uncertified Legendre c_n^+ estimate dashed
    # with a light band rather than presenting it as a certified bracket.
    widths.plot_gelfand_bounds(
        ax,
        cN,
        cn_lo,
        cn,
        r["cn_certified"],
        color="C1",
    )
    nn = np.arange(2, n_used)
    ax.loglog(
        nn,
        cn_mid[3] * (nn / cN[3]) ** (-(s - 0.5)),
        "k--",
        lw=1,
        label=r"$\propto n^{-(s-1/2)}$",
    )
    ax.set_xlabel(r"$m$ (points) $/$ $n$ (width)")
    ax.set_ylabel(r"$\|\cdot\|_\infty$ width")
    ax.set_title(rf"Legendre RKHS, $s={s:g}$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return r


def comparison_figure(
    smoothness_values=SMOOTHNESS_VALUES,
    n_trunc=MERCER_TRUNCATION,
    max_iter=1000,
    sel_grid=CANDIDATE_GRID_SIZE,
    cn_grid=5000,
    n_cap=1000,
    edge_ladder=ENDPOINT_LADDER_SIZE,
    compress_irls=True,
    out="figures/legendre.png",
):
    """Generate the paper's multi-panel Legendre comparison figure.

    The returned dictionary prefixes every numerical result by its smoothness (for example,
    ``s2_g`` and ``s3_cn``) so the publication runner can persist both panels in one NPZ.
    """
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
    for ax_i, s in zip(axes[0], smoothness_values):
        result = _rates_panel(
            ax_i,
            s=s,
            n_trunc=n_trunc,
            max_iter=max_iter,
            sel_grid=sel_grid,
            edge_ladder=edge_ladder,
            cn_grid=cn_grid,
            n_cap=n_cap,
            compress_irls=compress_irls,
        )
        tag = f"s{s:g}".replace(".", "p")
        combined.update({f"{tag}_{key}": value for key, value in result.items()})
        print(
            f"  panel {tag}: n_used={result['n_used']}, "
            f"sampling grid={sel_grid}"
        )

    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  figure saved -> {out}")
    return combined


def points_figure(
    smoothness_values=SMOOTHNESS_VALUES,
    m=POINT_DESIGN_SIZE,
    grid=CANDIDATE_GRID_SIZE,
    n_trunc=MERCER_TRUNCATION,
    out="figures/legendre_points.png",
):
    """Compare Legendre P-greedy designs across smoothness values at one size."""
    smoothness_values = tuple(float(s) for s in smoothness_values)
    m = int(m)
    if not smoothness_values:
        raise ValueError("smoothness_values must not be empty")
    if m < 1:
        raise ValueError("m must be positive")

    fig, axes = plt.subplots(
        1,
        len(smoothness_values),
        figsize=(5.0 * len(smoothness_values), 4.8),
        squeeze=False,
    )
    designs = {}
    for ax, s in zip(axes[0], smoothness_values):
        ker = LegendreMercerKernel(s=s, n_trunc=n_trunc)
        max_iter = int(np.ceil(1.2 * m))
        gr = PGreedy(ker, max_iter=max_iter, dtype=ker.dtype).fit(
            widths.box_grid(grid, 1, ker.dtype, kind="chebyshev")
        )
        if gr.n_ < m:
            plt.close(fig)
            raise RuntimeError(
                f"P-greedy stopped at {gr.n_} centers before the requested m={m} for s={s:g}"
            )

        n_shown = min(gr.n_, max_iter)
        centers = gr.ctrs_[:n_shown].reshape(-1).cpu().numpy()
        design = centers[:m]
        order = np.arange(1, n_shown + 1)
        ax.scatter(centers, order, c=order, cmap="viridis", s=16, zorder=2)
        ax.axhline(m, color="C3", lw=1.2, ls="--", zorder=3)
        ax.plot(
            design,
            np.full(m, m),
            "|",
            color="C3",
            ms=9,
            mew=1.4,
            label=rf"first $m={m}$ centers",
            zorder=4,
        )
        ax.set_xlabel(r"center location $x_i$")
        ax.set_ylabel("selection step")
        ax.set_title(rf"Legendre kernel: $s={s:g}$, $m={m}$")
        ax.set_xlim(-1.03, 1.03)
        ax.set_ylim(0, 1.03 * n_shown)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        designs[f"s{s:g}"] = gr
        print(
            f"  s={s:g}, m={m:3d} design highlighted; "
            f"selections shown through {n_shown}"
        )

    fig.suptitle("Endpoint clustering in Legendre P-greedy designs")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  figure saved -> {out}")
    return designs


if __name__ == "__main__":
    comparison_figure()
    points_figure()

"""
Sampling numbers vs Gelfand widths for the MATERN kernel, in d=1 and d=3.

The Legendre experiment (legendre.py) repeated for a Matern nu=3/2 RKHS -- a bounded kernel with
regularly-varying (algebraic) singular numbers, the regime the comparison theorem of
MAIN_Sampl_vs_Gelfand.tex covers.  For each dimension the two widths g_m^lin and c_n run parallel
and the ratio g_m/c_n stays BOUNDED (no sqrt(n)), whereas the general KPUU bound only gives
g_{2n} <= C sqrt(n) c_n.  The theorem has no dimension in its hypotheses, and the bounded ratio
indeed persists in d=3 (Matern is stationary, so it drops into higher d unchanged).

Two figures: matern.png (the g_m^lin vs c_n comparison, d=1 and d=3) and
matern_points.png (the P-greedy design in d=1: quasi-uniform gap-bisection -- the structure
behind the staircase g_m curve, contrast the endpoint-clustered Legendre design).
"""

from __future__ import annotations
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import MaternKernel
from greedy import design_figure
import widths

torch.manual_seed(0)


def rates_figure():
    specs = [
        (
            1,
            MaternKernel(nu=1.5, ell=0.3),
            dict(m_max=460, sel_grid=16000, cn_grid=2000, n_cap=300),
        ),
        (
            3,
            MaternKernel(nu=1.5, ell=0.6),
            dict(m_max=400, sel_grid=20000, cn_grid=8000, n_cap=140, refine_iters=100),
        ),
    ]

    results = []
    for d, ker, cfg in specs:
        r = widths.sampling_vs_gelfand(ker, d=d, **cfg)
        g_at_cN = np.exp(
            np.interp(np.log(r["cN"]), np.log(r["m_list"]), np.log(r["g"]))
        )
        r["ratio"] = g_at_cN / r["cn"]  # ratio at the (upper) c_n
        results.append((rf"Mat\'ern $\nu=3/2$, $d={d}$", r))
        print(
            f"Matern d={d}:  n_used={r['n_used']:4d}   "
            f"median g_m/c_n = {np.median(r['ratio']):.2f}  "
            f"(range {r['ratio'].min():.2f}-{r['ratio'].max():.2f})"
        )

    fig, axes = plt.subplots(
        1, len(specs), figsize=(5.0 * len(specs), 4.2), squeeze=False
    )
    for j, (label, r) in enumerate(results):
        # --- the two widths ---
        ax = axes[0, j]
        ax.loglog(
            r["m_list"],
            r["g"],
            "o-",
            ms=3.5,
            color="C0",
            label=r"$g_m^{\mathrm{lin}}$ (sampling)",
        )
        ax.fill_between(
            r["cN"],
            r["cn_lo"],
            r["cn"],
            color="C1",
            alpha=0.3,
            lw=0,
            label=r"$c_n\in[c_n^-,c_n^+]$ (Gelfand $d_n$)",
        )
        ax.loglog(r["cN"], r["cn_lo"], "-", color="C1", lw=0.8)
        ax.loglog(r["cN"], r["cn"], "-", color="C1", lw=0.8)
        ax.set_xlabel(r"$m$ (points) $/$ $n$ (width)")
        ax.set_ylabel(r"$\|\cdot\|_\infty$ width")
        ax.set_title(f"{label}\nmedian $g_m/c_n = {np.median(r['ratio']):.2f}$")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        r"Mat\'ern $g_m^{\mathrm{lin}}$ vs Gelfand widths $c_n$: the ratio stays "
        r"bounded (no $\sqrt{n}$) in $d=1$ and $d=3$"
    )
    fig.tight_layout()
    fig.savefig("matern.png", dpi=130, bbox_inches="tight", pad_inches=0.02)
    print("figure saved -> matern.png")


def points_figure(grid=4000):
    """P-greedy design for the Matern kernel (d=1): quasi-uniform gap-bisection (the staircase
    g_m signature of a stationary kernel).  Uniform candidate grid (no endpoint preference).
    """
    ker = MaternKernel(nu=1.5, ell=0.3)
    design_figure(
        ker,
        widths.box_grid(grid, 1, ker.dtype, kind="uniform"),
        r"Mat\'ern $\nu=3/2$",
        "quasi-uniform",
        "matern_points.png",
    )


if __name__ == "__main__":
    rates_figure()
    points_figure()

"""
Supplemental experiment: estimated sampling numbers vs Gelfand-width lower/upper values for the
MATERN kernel, in d=1 and d=3.  This experiment is not part of the manuscript's numerical section.

The Legendre experiment (legendre.py) repeated for a Matern nu=3/2 RKHS -- a bounded kernel with
regularly-varying (algebraic) singular numbers, the regime the comparison theorem of
the companion manuscript covers.  For each dimension the estimated sampling numbers and c_n run
parallel, with no observed sqrt(n) growth.  Here g_m^lin is numerically estimated from one P-greedy
design, giving an upper surrogate rather than the global point-set infimum.  Matern is stationary,
so the same implementation applies unchanged in d=3.

Two figures: matern.png (the estimated g_m^lin vs c_n comparison, d=1 and d=3) and
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


def rates_figure(compress_irls=True):
    specs = [
        (
            1,
            MaternKernel(nu=1.5, ell=0.3),
            dict(
                m_max=460,
                sel_grid=16000,
                cn_grid=2000,
                n_cap=300,
                certify_evals=500_000,
            ),
        ),
        (
            3,
            MaternKernel(nu=1.5, ell=0.6),
            dict(
                m_max=400,
                sel_grid=20000,
                cn_grid=8000,
                n_cap=140,
                refine_iters=100,
                certify_tol=0.02,
                certify_evals=500_000,
            ),
        ),
    ]

    results = []
    for d, ker, cfg in specs:
        r = widths.sampling_vs_gelfand(ker, d=d, compress_irls=compress_irls, **cfg)
        g_at_cN = np.exp(
            np.interp(np.log(r["cN"]), np.log(r["m_list"]), np.log(r["g"]))
        )
        r["ratio_lo"] = g_at_cN / r["cn"]
        r["ratio_hi"] = g_at_cN / r["cn_lo"]
        r["ratio_kind"] = ("certified ratio interval" if r["cn_certified"].all()
                           else "indicator range")
        results.append((rf"Mat\'ern $\nu=3/2$, $d={d}$", r))
        print(
            f"Matern d={d}:  n_used={r['n_used']:4d}   "
            f"median estimated g_m/c_n {r['ratio_kind']} = [{np.median(r['ratio_lo']):.2f}, "
            f"{np.median(r['ratio_hi']):.2f}]  "
            f"({int(r['cn_certified'].sum())}/{len(r['cN'])} uppers certified)"
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
            label=r"$g_m^{\mathrm{lin}}$ (P-greedy estimate)",
        )
        widths.plot_gelfand_bounds(
            ax,
            r["cN"],
            r["cn_lo"],
            r["cn"],
            r["cn_certified"],
            color="C1",
        )
        ax.set_xlabel(r"$m$ (points) $/$ $n$ (width)")
        ax.set_ylabel(r"$\|\cdot\|_\infty$ width")
        ax.set_title(f"{label}\nmedian estimated $g_m/c_n$ {r['ratio_kind']} "
                     f"[{np.median(r['ratio_lo']):.2f}, {np.median(r['ratio_hi']):.2f}]")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        r"Mat\'ern estimated sampling numbers vs Gelfand widths: "
        r"no observed $\sqrt{n}$ growth in $d=1$ and $d=3$"
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

"""
Driver: assess the sampling-vs-Gelfand comparison from the companion manuscript on the
univariate Legendre Mercer RKHS (the kernel example is from arXiv:2103.11124, Section 7.3).

Plots, in the paper's notation:
  * g_m^lin       sampling numbers, numerically estimated by the grid-resolved power function of
                  the computed P-greedy design (an upper surrogate, not the exact infimum).
  * c_n           Gelfand widths d_n(K)_H, the numerical Kolmogorov width of the translates.
  * optional diagnostics: sqrt(n) c_n, sigma_n = sqrt(mu_n), and the n^-s rate guide
    (`diagnostic_overlays=True`).  The manuscript n^-(s-1/2) guide is always shown.

Result: the estimated sampling numbers and c_n run parallel on n^-(s-1/2), with no observed sqrt(n)
growth; this is numerical evidence for the comparison theorem, not an exact computation of g_m^lin.

Two figures: legendre.png (the estimated g_m^lin vs c_n comparison) and legendre_points.png (the
P-greedy point selection used for the estimate -- power-function snapshot + endpoint-clustered design).
s=2 keeps Pow^2 above the float64 cancellation floor out to large m; s>=4 would stop early
(~m=200) as Pow^2 -> 0 hits tol_p.
"""

from __future__ import annotations
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import LegendreMercerKernel
from greedy import design_figure
import widths

torch.manual_seed(0)


# Publication run: M=24000 keeps the omitted endpoint-energy scale below the smallest
# widths tracked near n=1000.  This Mercer truncation is specific to the Legendre feature
# expansion; similarly sized values in other drivers are sampling-node budgets, not M.

def rates_figure(
    s=2.0, n_trunc=24000, max_iter=1000, sel_grid=20000, cn_grid=5000, n_cap=1000,
    compress_irls=True,
    diagnostic_overlays=False,
):
    kernel = LegendreMercerKernel(s=s, n_trunc=n_trunc)
    r = widths.sampling_vs_gelfand(
        kernel, m_max=max_iter, sel_grid=sel_grid, cn_grid=cn_grid, n_cap=n_cap,
        compress_irls=compress_irls,
    )
    m_list, g, cN, cn, n_used = r["m_list"], r["g"], r["cN"], r["cn"], r["n_used"]
    cn_lo = r["cn_lo"]  # c_n^- is rigorous; c_n^+ is certified only where flagged
    cn_mid = np.sqrt(
        cn_lo * cn
    )  # geometric-mean representative for the manuscript rate guide

    # --- plot: estimated sampling numbers vs Gelfand widths ---
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
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
    if diagnostic_overlays:
        ax.loglog(
            cN,
            np.sqrt(cN) * cn_mid,
            "-",
            lw=1.3,
            color="0.55",
            label=r"$\sqrt{n}\,c_n$  (KPUU bound, Eq. 15 -- the closed gap)",
        )
        sig = kernel.sqrt_mu[:n_used].cpu().numpy()
        ax.loglog(
            np.arange(1, len(sig)),
            sig[1:],
            ".-",
            ms=2.5,
            color="C2",
            alpha=0.55,
            label=r"$\sigma_n=\sqrt{\mu_n}$  (singular numbers)",
        )
    nn = np.arange(2, n_used)
    ax.loglog(
        nn,
        cn_mid[3] * (nn / cN[3]) ** (-(s - 0.5)),
        "k--",
        lw=1,
        label=r"$\propto n^{-(s-1/2)}$",
    )
    if diagnostic_overlays:
        ax.loglog(nn, sig[2] * (nn / 2.0) ** (-s), "k:", lw=1, label=r"$\propto n^{-s}$")
    ax.set_xlabel(r"index  $m$ (sampling points)  $/$  $n$ (width)")
    ax.set_ylabel(r"uniform-norm width  ($\|\cdot\|_\infty$)")
    ax.set_title(
        rf"Estimated sampling numbers $g_m^{{\mathrm{{lin}}}}$ vs Gelfand widths $c_n$"
        rf"   (Legendre RKHS, $s={s:g}$)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("legendre.png", dpi=130, bbox_inches="tight", pad_inches=0.02)
    print(f"  figure saved -> legendre.png  (n_used={n_used})")
    return r


def points_figure(s=2.0, grid=4000):
    """P-greedy design for the Legendre kernel: endpoint-clustered (the structure behind the
    smooth g_m curve).  Chebyshev candidate grid (boundary-concentrated Mercer kernel).
    """
    ker = LegendreMercerKernel(s=s, n_trunc=4000)
    design_figure(
        ker,
        widths.box_grid(grid, 1, ker.dtype, kind="chebyshev"),
        rf"Legendre $s={s:g}$",
        "endpoint-clustered",
        "legendre_points.png",
    )


if __name__ == "__main__":
    rates_figure(s=2.0)
    points_figure(s=2.0)

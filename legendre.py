"""
Driver: assess the sampling-vs-Gelfand comparison of MAIN_Sampl_vs_Gelfand.tex on the
univariate Legendre Mercer RKHS (Section 7.3 of arXiv:2103.11124).

Plots, in the paper's notation:
  * g_m^lin       linear sampling numbers inf_{|P|=m} ||Pow_P||_inf, from the P-greedy design.
  * c_n           Gelfand widths d_n(K)_H, the numerical Kolmogorov width of the translates.
  * sqrt(n) c_n   the general KPUU bound g_{2n} <= C sqrt(n) c_n -- the sqrt(n) the theorem removes.
  * sigma_n = sqrt(mu_n)  singular numbers, plus rate guides n^-(s-1/2), n^-s.

Result: g_m^lin and c_n run parallel on n^-(s-1/2) with a BOUNDED ratio (no sqrt(n)), a full
half-power below sqrt(n) c_n -- Thm 'Carl-type comparison' on the example.

Two figures: legendre.png (the g_m^lin vs c_n comparison) and legendre_points.png (the P-greedy
point selection that produces g_m^lin -- power-function snapshot + endpoint-clustered design).
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


def rates_figure(
    s=2.0, n_trunc=24000, max_iter=1000, sel_grid=20000, cn_grid=5000, n_cap=1000
):
    kernel = LegendreMercerKernel(s=s, n_trunc=n_trunc)
    r = widths.sampling_vs_gelfand(
        kernel, m_max=max_iter, sel_grid=sel_grid, cn_grid=cn_grid, n_cap=n_cap
    )
    m_list, g, cN, cn, n_used = r["m_list"], r["g"], r["cN"], r["cn"], r["n_used"]
    cn_lo = r["cn_lo"]  # c_n bracket: cn_lo certified (lower), cn estimated (upper)
    cn_mid = np.sqrt(
        cn_lo * cn
    )  # geometric-mean representative (for the rate/KPUU curves)
    # singular numbers sigma_n = sqrt(mu_n) of the embedding (exact, Mercer kernel)
    sig = kernel.sqrt_mu[:n_used].cpu().numpy()

    # --- plot: sampling numbers vs Gelfand widths, paper notation ---
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.loglog(
        m_list,
        g,
        "o-",
        ms=3.5,
        color="C0",
        label=r"$g_m^{\mathrm{lin}}$  (P-greedy sampling numbers)",
    )
    # c_n as its bracket [c_n^-, c_n^+], not a curve; band width = honest uncertainty.
    # c_n^- certified (weak duality); c_n^+ an off-grid sup estimate -- Legendre has no finite
    # RKHS modulus at +-1, so no branch-and-bound certificate; the endpoint-laddered grid +
    # deterministic endpoint sweep resolve the boundary spike (see widths.gelfand_widths).
    ax.fill_between(
        cN,
        cn_lo,
        cn,
        color="C1",
        alpha=0.3,
        lw=0,
        label=r"$c_n\in[c_n^-,c_n^+]$  (Gelfand width $d_n$)",
    )
    ax.loglog(cN, cn_lo, "-", color="C1", lw=0.8)  # bracket edges (visible when tight)
    ax.loglog(cN, cn, "-", color="C1", lw=0.8)
    ax.loglog(
        cN,
        np.sqrt(cN) * cn_mid,
        "-",
        lw=1.3,
        color="0.55",
        label=r"$\sqrt{n}\,c_n$  (KPUU bound, Eq. 15 -- the closed gap)",
    )
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
    ax.loglog(nn, sig[2] * (nn / 2.0) ** (-s), "k:", lw=1, label=r"$\propto n^{-s}$")
    ax.set_xlabel(r"index  $m$ (sampling points)  $/$  $n$ (width)")
    ax.set_ylabel(r"uniform-norm width  ($\|\cdot\|_\infty$)")
    ax.set_title(
        rf"Sampling numbers $g_m^{{\mathrm{{lin}}}}$ vs Gelfand widths $c_n$"
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

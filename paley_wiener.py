"""
Estimated sampling numbers vs Gelfand-width lower/upper values for the Paley-Wiener kernel, in float64.

    K_c(s,t) = sin(c(s-t))/(pi(s-t)),   s,t in [-1,1],   K(x,x) = c/pi,   N_eff = 2c/pi.

The comparison theorem in the companion manuscript assumes regularly-varying (algebraic) singular
numbers (legendre.py / matern.py).  The band-limited kernel is the opposite regime: its sigma_k are
the Slepian eigenvalues -- flat ~1 up to N_eff, then a super-exponential cliff.  The estimated
sampling numbers g_m^lin and c_n are both ~flat and then fall off the SAME cliff at N_eff: below
it the curves numerically track one another (on a kernel outside the theorem's
hypotheses); at n ~ N_eff both hit the float64 cancellation floor tau ~ sqrt(eps)*sqrt(K(x,x)) ~ 1e-6,
because the band-limited Gram has numerical rank ~ N_eff (past the cliff the recovered values ARE the
floor, not the true widths).

Recovering the true, super-exponentially small widths past the cliff would need an arbitrary-
precision (mpmath) rerun of the width minimax; the float64 comparison below N_eff is the result
of interest here (the floored values past the cliff carry no information).

Two figures, three bandwidths N_eff in {20,40,80} (the cliff marches right in proportion to c):
  figures/paley_wiener.png        -- estimated g_m^lin, float64 c_n lower/upper values, and sigma_n, each falling off the
                             N_eff cliff into the float64 floor.
  figures/paley_wiener_points.png -- the P-greedy design: quasi-uniform at Nyquist spacing 2/N_eff up to N_eff.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import PaleyWienerSincKernel
from greedy import PGreedy
import widths

torch.manual_seed(0)
FIGURES_DIR = Path("figures")

# Estimated g_m^lin / c_n config per bandwidth.  The c_n upper value is a branch-and-bound certificate
# (sinc has a dist_bound modulus); at/past the N_eff cliff, where the widths ARE the float64
# floor, the certificate cannot converge and gelfand_widths warns + falls back to the dense-scan
# estimate there -- expected, the floored values carry no information to certify.
SPECS = [
    (20, dict(m_max=70, sel_grid=8000, cn_grid=2000, n_cn=28, n_cap=34)),
    (40, dict(m_max=110, sel_grid=8000, cn_grid=3000, n_cn=32, n_cap=56)),
    (80, dict(m_max=140, sel_grid=10000, cn_grid=4000, n_cn=34, n_cap=95)),
]


def rates_figure(compress_irls=True):
    results = []
    for n_eff, cfg in SPECS:
        ker = PaleyWienerSincKernel(n_eff=n_eff)
        r = widths.sampling_vs_gelfand(ker, d=1, compress_irls=compress_irls, **cfg)
        r["sigma"] = ker.singular_numbers(
            r["n_used"]
        )  # float64 sigma_n (floors past N_eff)
        r["n_eff"], r["diag"] = ker.n_eff, ker.diag_val

        g_at_cN = np.exp(
            np.interp(np.log(r["cN"]), np.log(r["m_list"]), np.log(r["g"]))
        )
        r["g_at_cN"] = g_at_cN
        r["ratio_lo"] = g_at_cN / r["cn"]
        r["ratio_hi"] = g_at_cN / r["cn_lo"]

        # reliability mask: both widths above the band-limited-Gram floor
        tau = 20.0 * np.sqrt(np.finfo(np.float64).eps) * np.sqrt(r["diag"])
        rel = (g_at_cN > tau) & (r["cn"] > tau) & (r["cn_lo"] > tau)
        r["rel"], r["tau"] = rel, tau
        results.append((n_eff, r))

        rr_lo = r["ratio_lo"][rel]
        rr_hi = r["ratio_hi"][rel]
        ratio_kind = ("certified ratio interval" if r["cn_certified"][rel].all()
                      else "numerical indicator range")
        print(
            f"N_eff={n_eff:3d}  reliable n<= {int(r['cN'][rel].max()) if rel.any() else 0:3d}  "
            f"median estimated g_m/c_n {ratio_kind} = [{np.median(rr_lo):.2f}, {np.median(rr_hi):.2f}]  "
            f"({int(r['cn_certified'][rel].sum())}/{int(rel.sum())} reliable uppers certified)"
            if rel.any()
            else f"N_eff={n_eff}: no reliable float64 window"
        )

    # ---------------- figure: widths x 3 bandwidths ----------------
    fig, axes = plt.subplots(
        1, len(SPECS), figsize=(5.4 * len(SPECS), 4.5), squeeze=False
    )
    for j, (n_eff, r) in enumerate(results):
        ne = r["n_eff"]
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
        ax.loglog(
            np.arange(1, len(r["sigma"])),
            r["sigma"][1:],
            ":",
            color="C2",
            lw=1.1,
            alpha=0.8,
            label=r"$\sigma_n=\sqrt{\lambda_n}$ (prolate)",
        )
        ax.axvline(ne, color="0.4", ls="--", lw=1.0)
        ax.text(
            ne * 1.02,
            r["tau"] * 3,
            r"$N_{\mathrm{eff}}=%g$" % ne,
            rotation=90,
            va="bottom",
            fontsize=8,
            color="0.35",
        )
        ax.axhspan(1e-20, r["tau"], color="0.85", alpha=0.6, lw=0)
        ax.text(
            0.97,
            0.03,
            f"float64 floor ~{r['tau']:.0e}",
            transform=ax.transAxes,
            fontsize=7,
            color="0.4",
            ha="right",
            va="bottom",
        )
        ax.set_ylim(0.3 * r["tau"], 8.0)
        ax.set_xlabel(r"$m$ (points) $/$ $n$ (width)")
        ax.set_ylabel(r"uniform-norm width $\|\cdot\|_\infty$")
        ax.set_title(
            rf"Paley-Wiener $K_c$, $c={ne * np.pi / 2:.1f}$  ($N_{{\mathrm{{eff}}}}={ne}$)"
        )
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        r"Band-limited kernel: estimated sampling numbers and Gelfand lower/upper values "
        r"fall off the Slepian cliff at $N_{\mathrm{eff}}=2c/\pi$, into the float64 floor",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "paley_wiener.png", dpi=130, bbox_inches="tight", pad_inches=0.02)
    print("figure saved -> figures/paley_wiener.png")


# ---------------------------------------------------------------------------------------------
# P-greedy point selection: the band-limited design is quasi-uniform at the Nyquist spacing
# 2/N_eff; the n=N_eff line is the saturated threshold design.  Candidate measure is UNIFORM
# (stationary kernel: no endpoint preference).
# ---------------------------------------------------------------------------------------------
def points_figure(grid=6000):
    n_effs = [20, 40, 80]
    fig, axes = plt.subplots(
        1, len(n_effs), figsize=(5.4 * len(n_effs), 4.8), squeeze=False
    )
    for j, n_eff in enumerate(n_effs):
        ker = PaleyWienerSincKernel(n_eff=n_eff)
        n_thr = int(round(n_eff))
        n_design = int(
            round(1.2 * n_eff)
        )  # just past N_eff (stay above the float64 floor)
        gr = PGreedy(ker, max_iter=n_design, dtype=ker.dtype).fit(
            widths.box_grid(grid, 1, ker.dtype, kind="uniform", domain=ker.domain)
        )
        ctrs = gr.ctrs_.reshape(-1).cpu().numpy()
        thr = np.sort(ctrs[:n_thr])
        ideal = -1.0 + (np.arange(n_thr) + 0.5) * (2.0 / n_thr)
        dev = np.abs(thr - ideal)
        dx = np.diff(thr)

        ax = axes[0, j]
        ax.scatter(
            ctrs,
            np.arange(1, len(ctrs) + 1),
            c=np.arange(1, len(ctrs) + 1),
            cmap="viridis",
            s=14,
            zorder=2,
        )
        ax.axhline(n_thr, color="C3", lw=1.2, ls="--", zorder=3)
        ax.plot(
            thr,
            np.full_like(thr, n_thr),
            "|",
            color="C3",
            ms=9,
            mew=1.4,
            zorder=4,
            label=rf"threshold design at $n=N_{{\mathrm{{eff}}}}={n_eff}$",
        )
        ax.set_xlabel(r"center position $x \in [-1,1]$")
        ax.set_ylabel(r"selection order $n$")
        ax.set_title(
            rf"$N_{{\mathrm{{eff}}}}={n_eff}$: quasi-uniform, max dev {dev.max():.3f} vs Nyquist"
        )
        ax.set_xlim(-1.03, 1.03)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        print(
            f"N_eff={n_eff:3d}  threshold design n={n_thr}:  mean spacing {dx.mean():.4f} "
            f"(Nyquist {2.0/n_eff:.4f}), max dev from uniform {dev.max():.4f}"
        )

    fig.suptitle(
        r"P-greedy design for the band-limited (Paley-Wiener) kernel: the $n=N_{\mathrm{eff}}$ "
        r"line holds every point found up to the threshold"
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES_DIR / "paley_wiener_points.png",
        dpi=130,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    print("figure saved -> figures/paley_wiener_points.png")


if __name__ == "__main__":
    rates_figure()
    points_figure()

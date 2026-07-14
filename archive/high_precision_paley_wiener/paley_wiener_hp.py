"""
Sampling numbers vs Gelfand widths for the BAND-LIMITED (Paley-Wiener / prolate) kernel, with c_n
pushed PAST the float64 cliff by a high-precision (mpmath) computation.

    K_c(s,t) = sin(c(s-t))/(pi(s-t)),   s,t in [-1,1],   K(x,x) = c/pi,   N_eff = 2c/pi.

The comparison theorem of MAIN_Sampl_vs_Gelfand.tex assumes regularly-varying (algebraic) singular
numbers (legendre.py / matern.py).  The band-limited kernel is the opposite regime: its sigma_k are
the Slepian eigenvalues -- flat ~1 up to N_eff, then a super-exponential cliff.  Its Gram has
numerical rank ~ N_eff, so widths.gelfand_widths floors at ~sqrt(eps)*sqrt(K(x,x)) ~ 1e-6 once n
reaches N_eff (past the cliff it returns the floor, not the true widths).  widths.gelfand_widths_hp
runs the same minimax in mpmath on the exact prolate mode coordinates and recovers the true c_n
bracket many orders lower (tight, ratio ~1.3); where both are reliable they agree to a few percent
(independent cross-check), and at the cliff the hp bracket is tighter.

Two figures, three bandwidths N_eff in {20,40,80} (the cliff marches right in proportion to c):
  paley_wiener.png        -- g_m^lin, the float64 c_n bracket, the high-precision c_n bracket
                             continuing past the floor, and sigma_n.  (g_m is NOT lifted past the
                             cliff -- that needs an hp greedy -- so the hp extends the WIDTH panel
                             past the point where g_m can follow.)
  paley_wiener_points.png -- the P-greedy design: quasi-uniform at Nyquist spacing 2/N_eff up to N_eff.

The hp bracket (pure-Python mpmath eigensolves) is the slow part; it is cached to paley_wiener_hp.npz
and reused unless the n-lists change (recompute=True forces).
"""

from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# hp classes live in this archive folder; the float64 pipeline (widths, greedy) is imported from
# the parent package -- see README.md for the PYTHONPATH build recipe.
from kernels_hp import PaleyWienerSincKernelHP as PaleyWienerSincKernel
from widths_hp import gelfand_widths_hp
from greedy import PGreedy
import widths

torch.manual_seed(0)

# float64 g_m^lin / c_n config per bandwidth (as in the former paley_wiener.py)
SPECS = [
    (20, dict(m_max=70, sel_grid=8000, cn_grid=2000, n_cn=28, n_cap=34, rule="hedge")),
    (40, dict(m_max=110, sel_grid=8000, cn_grid=3000, n_cn=32, n_cap=56, rule="hedge")),
    (
        80,
        dict(m_max=140, sel_grid=10000, cn_grid=4000, n_cn=34, n_cap=95, rule="hedge"),
    ),
]
# n-abscissae for the high-precision bracket: from below each cliff to well past it (down to the
# depth where the tight hp bracket is worth showing -- ~1e-16 for N_eff=40, the showcase).
HP_NLISTS = {
    20: list(range(12, 31, 2)),
    40: list(range(32, 69, 4)) + [70],
    80: list(range(66, 91, 3)),
}
CACHE = "paley_wiener_hp.npz"


def compute_hp(recompute: bool = False):
    """High-precision c_n brackets for every bandwidth, cached to paley_wiener_hp.npz."""
    if not recompute and os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        cached = d["nlists"].item()
        if cached == HP_NLISTS:
            print(f"hp bracket: loaded cache {CACHE}")
            return d["hp"].item()
        print("hp bracket: n-lists changed, recomputing")
    hp = {}
    for n_eff, nlist in HP_NLISTS.items():
        print(f"hp bracket N_eff={n_eff} (n={nlist[0]}..{nlist[-1]}) ...", flush=True)
        ker = PaleyWienerSincKernel(
            n_eff=n_eff
        )  # supplies the arbitrary-precision kfun_mp
        hp[n_eff] = gelfand_widths_hp(ker, nlist, verbose=True)
    np.savez(
        CACHE, hp=np.array(hp, dtype=object), nlists=np.array(HP_NLISTS, dtype=object)
    )
    print(f"hp bracket: saved cache {CACHE}")
    return hp


def rates_figure(recompute_hp: bool = False):
    hp = compute_hp(recompute=recompute_hp)

    results = []
    for n_eff, cfg in SPECS:
        ker = PaleyWienerSincKernel(n_eff=n_eff)
        r = widths.sampling_vs_gelfand(ker, d=1, **cfg)
        r["sigma"] = ker.singular_numbers(
            r["n_used"]
        )  # float64 sigma_n (floors past N_eff)
        r["n_eff"], r["diag"] = ker.n_eff, ker.diag_val

        g_at_cN = np.exp(
            np.interp(np.log(r["cN"]), np.log(r["m_list"]), np.log(r["g"]))
        )
        r["g_at_cN"] = g_at_cN
        r["ratio"] = g_at_cN / r["cn"]

        # float64 reliability mask: both widths above the band-limited-Gram floor
        tau = 20.0 * np.sqrt(np.finfo(np.float64).eps) * np.sqrt(r["diag"])
        rel = (g_at_cN > tau) & (r["cn"] > tau) & (r["cn_lo"] > tau)
        r["rel"], r["tau"] = rel, tau
        results.append((n_eff, r, hp[n_eff]))

        rr = r["ratio"][rel]
        print(
            f"N_eff={n_eff:3d}  reliable n<= {int(r['cN'][rel].max()) if rel.any() else 0:3d}  "
            f"median g_m/c_n = {np.median(rr):.2f}   hp c_n reaches {hp[n_eff]['c_lo'].min():.1e}"
            if rel.any()
            else f"N_eff={n_eff}: no reliable float64 window"
        )

    # ---------------- figure: widths x 3 bandwidths ----------------
    fig, axes = plt.subplots(
        1, len(SPECS), figsize=(5.4 * len(SPECS), 4.5), squeeze=False
    )
    for j, (n_eff, r, h) in enumerate(results):
        ne = r["n_eff"]

        # --- g_m, float64 c_n bracket, HIGH-PRECISION c_n bracket, sigma_n ---
        ax = axes[0, j]
        ax.loglog(
            r["m_list"],
            r["g"],
            "o-",
            ms=3.5,
            color="C0",
            label=r"$g_m^{\mathrm{lin}}$ (P-greedy sampling)",
        )
        ax.fill_between(
            r["cN"],
            r["cn_lo"],
            r["cn"],
            color="0.5",
            alpha=0.3,
            lw=0,
            label=r"$c_n$ float64 grid $[c_n^-,c_n^+]$",
        )
        ax.loglog(r["cN"], r["cn_lo"], "-", color="0.4", lw=0.7)
        ax.loglog(r["cN"], r["cn"], "-", color="0.4", lw=0.7)
        ax.fill_between(
            h["n"],
            h["c_lo"],
            h["c_hi"],
            color="C1",
            alpha=0.35,
            lw=0,
            label=r"$c_n$ high precision $[c_n^-,c_n^+]$",
        )
        ax.loglog(h["n"], h["c_lo"], "o-", color="C1", ms=2.5, lw=1.0)
        ax.loglog(h["n"], h["c_hi"], "o-", color="C1", ms=2.5, lw=1.0)
        ax.loglog(
            h["n"],
            h["sigma"],
            ":",
            color="C2",
            lw=1.1,
            alpha=0.8,
            label=r"$\sigma_n=\sqrt{\lambda_n}$ (prolate)",
        )
        ax.axvline(ne, color="0.4", ls="--", lw=1.0)
        ax.text(
            ne * 1.02,
            ax.get_ylim()[0] * 4,
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
        ax.set_ylim(0.3 * h["c_lo"].min(), 8.0)
        ax.set_xlabel(r"$m$ (points) $/$ $n$ (width)")
        ax.set_ylabel(r"uniform-norm width $\|\cdot\|_\infty$")
        ax.set_title(
            rf"Paley-Wiener $K_c$, $c={ne * np.pi / 2:.1f}$  ($N_{{\mathrm{{eff}}}}={ne}$)"
        )
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        r"Band-limited (Paley-Wiener) kernel: high precision recovers the true Gelfand "
        r"widths $c_n$ past the $N_{\mathrm{eff}}=2c/\pi$ float64 cliff",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("paley_wiener.png", dpi=130, bbox_inches="tight", pad_inches=0.02)
    print("figure saved -> paley_wiener.png")


# ---------------------------------------------------------------------------------------------
# P-greedy point selection: the band-limited design is quasi-uniform at the Nyquist spacing
# 2/N_eff; the n=N_eff line is the saturated threshold design.  Candidate measure is UNIFORM
# (stationary kernel: no endpoint preference).  (Distinct 3-bandwidth layout, so not design_figure.)
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
            "o",
            color="C3",
            ms=5,
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
    fig.savefig(
        "paley_wiener_points.png", dpi=130, bbox_inches="tight", pad_inches=0.02
    )
    print("figure saved -> paley_wiener_points.png")


if __name__ == "__main__":
    rates_figure()
    points_figure()

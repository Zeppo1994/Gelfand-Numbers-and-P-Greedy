r"""
Periodic mixed-smoothness Sobolev RKHS  H^m_mix([0,1]^d)  (Berlinet-Thomas-Agnan
p.318, tensorized): the d-fold tensor of  k_1(s,t)=1+(-1)^{m-1}/(2m)! B_{2m}(|s-t|).
For moderate d and smoothness m = 1, 2, 3, two figures:

  periodic_mixed.png -- sampling numbers g_n^lin, numerically estimated by one P-greedy design
      (a constructive upper surrogate), vs n on log-log axes, against the Gelfand-width
      benchmark rate c_n ~ n^{-(m-1/2)} (log n)^{(d-1)m}.  Its constant is fitted by a
      single least-squares offset over the asymptotic tail (no per-point tuning); the
      empirical slope is printed next to the predicted -(m-1/2).
  periodic_mixed_points.png -- for d=2, the raw points the greedy places in [0,1]^2.

Notation: g_n^lin is the sampling-number estimate (n = #nodes), while m is the mixed
SMOOTHNESS of H^m_mix.  The estimate is an upper surrogate, not a computation of the exact
point-set infimum; c_n is the Gelfand width as everywhere.

Run:  python periodic_mixed.py
Precision: all Gelfand-width computations and the m>=2 P-greedy curves use float64.  The
m=1 P-greedy curve uses float32: it stays safely above that dtype's cancellation floor and
was validated against float64 to <0.6%, while providing about a 16x GPU speedup.

"""

from __future__ import annotations
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import PeriodicSobolevMixedKernel
from greedy import PGreedy
import widths

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def grid01(
    n: int, d: int, dtype: torch.dtype = torch.float64, device: str = DEVICE
) -> torch.Tensor:
    """Candidate nodes in the torus [0,1)^d where the greedy picks and the sup power is
    measured.  1D: uniform periodic grid (endpoint dropped).  d>1: scrambled Sobol (more
    even coverage than i.i.d. uniform, so a finite grid resolves the sup power better).
    Placed on `device` so the greedy runs on GPU when available."""
    if d == 1:
        g = torch.linspace(0, 1, n + 1, dtype=dtype)[:-1].reshape(-1, 1)
    else:
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=0)
        g = eng.draw(n).to(dtype)
    return g.to(device)


def greedy_curve(m: int, d: int, n_nodes: int, sel_grid: int):
    """Run the strong P-greedy and return (g_full, centers_in_order, n_used), where
    g_full[k] = sup_x Pow_k(x) is the worst-case L_inf error with k greedy nodes."""
    # dtype: float32 for m=1 is ~16x faster on GPU (-> many more nodes) and safe because
    # the slow g_n~n^{-1/2} decay stays above float32's power-cancellation floor (~1e-3)
    # to n~1e4.  m>=2 dives past it early (m=2 breaks near n~50), so keep float64.  (The
    # Gelfand widths c_n always use float64, separately.)
    prec = torch.float32 if m == 1 else torch.float64
    ker = PeriodicSobolevMixedKernel(m=m, d=d, dtype=prec, device=DEVICE)
    gr = PGreedy(ker, max_iter=n_nodes, dtype=prec).fit(
        grid01(sel_grid, d, prec, DEVICE)
    )
    return (gr.g_curve().cpu().double().numpy(), gr.ctrs_.cpu().double().numpy(), gr.n_)


def bench_rate(n, m, d):
    """The Gelfand-width benchmark rate  n^{-(m-1/2)} (log n)^{(d-1)m}  for H^m_mix."""
    n = np.asarray(n, float)
    return n ** (-(m - 0.5)) * np.log(n) ** ((d - 1) * m)


def fit_const(ns, y, m, d):
    """Least-squares vertical offset in log-space matching the benchmark rate to y over
    the whole range: log C = mean(log y - log rate).  Fitted to c_n (which the rate
    describes), not g_n."""
    logC = np.mean(np.log(y) - np.log(bench_rate(ns, m, d)))
    return np.exp(logC)


# --------------------------------------------------------------------------- #
#  (1)+(2)  Estimated sampling numbers g_n^lin vs the NUMERICAL Gelfand widths c_n
#
#  The theoretical rate n^{-(m-1/2)}(log n)^{(d-1)m} is only asymptotic: at reachable n
#  its (log n)^{(d-1)m} factor is far from developed (peaks near n = e^{(d-1)m/(m-1/2)}),
#  so a line fitted to a finite window bends away from the data in higher d.  The honest,
#  dimension-robust numerical comparison is the sampling-number estimate against the Gelfand-width
#  lower/upper band from widths.gelfand_widths.  The rate line is kept as a light guide, fitted to the
#  numerical upper width value.
# --------------------------------------------------------------------------- #
def rates_figure(compress_irls=True, diagnostic_overlays=False):
    # per case: (d, {m: greedy nodes}, greedy grid, c_n grid, max n for c_n).
    # Greedy budgets (GPU): m=1 (float32, cheap, floor-free) runs furthest on a fine grid;
    # m=2 (float64) is bandwidth-limited but reaches several thousand; m=3 is float64-floor-
    # limited near n~600.  c_n stays on CPU (its small fp64 eighs + refine beat GPU),
    # extended as far as the O(N^3) eigh affords, with cn_grid ~ 5*n_cap for the n<~N/5
    # lower/upper reliability.
    cases = [
        (2, {1: 12000, 2: 5000, 3: 1000}, 120000, 4500, 900),
        (3, {1: 10000, 2: 4000}, 80000, 4500, 900),
    ]
    colors = {1: "C0", 2: "C1", 3: "C2"}
    fig, axes = plt.subplots(
        1, len(cases), figsize=(7.0 * len(cases), 4.7), squeeze=False
    )

    for j, (d, m_nodes, sel, cn_grid, n_cap) in enumerate(cases):
        top = axes[0, j]
        print(f"\n=== d={d} ===  (greedy grid {sel}, c_n grid {cn_grid}, dev={DEVICE})")
        Xc = grid01(
            cn_grid, d, torch.float64, "cpu"
        )  # Gelfand widths: CPU float64 (scipy refine)
        for m, n_nodes in m_nodes.items():
            g, _, n_used = greedy_curve(m, d, n_nodes, sel)
            gn = np.arange(len(g))
            # Compare against c_n only where g_n stays above the greedy's float64
            # cancellation floor (~1e-6); below it the power update reads g_n spuriously
            # low and g_n/c_n dips unphysically under 1.  Full curve still plotted; only
            # the ratio range is trimmed (matters for m=3, which hits the floor by n~200).
            above = g > 2e-6
            n_rel = int(gn[above][-1]) if above.any() else n_used - 1
            ns_c = widths.log_spaced_ints(min(n_cap, n_used - 1, n_rel), 14)
            cn_up, cn_lo, cn_info = widths.gelfand_widths(
                PeriodicSobolevMixedKernel(m=m, d=d, dtype=torch.float64, device="cpu"),
                Xc,
                ns_c,
                n_iter=20,
                refine=True,
                refine_starts=32,
                refine_iters=40,
                return_info=True,
                compress_irls=compress_irls,
            )
            cn_up, cn_lo = cn_up.numpy(), cn_lo.numpy()
            g_at_c = np.interp(ns_c, gn, g)  # g_n sampled at the c_n abscissae
            certified = cn_info["upper_certified"]
            ratio_kind = ("certified ratio interval" if certified.all()
                          else "numerical indicator range")
            ratio_lo = g_at_c / cn_up
            ratio_hi = g_at_c / cn_lo
            col = colors[m]

            # asymptotic guide: the rate only makes sense past the hump of its log factor
            # at n_pk = exp((d-1)m/(m-1/2)); drawing from n=2 shoots it over the data and
            # biases C.  So fit C on the tail n>=n_start and draw from there to the curve end.
            n_pk = np.exp((d - 1) * m / (m - 0.5)) if (d - 1) * m else 2.0
            n_start = max(2.0 * n_pk, ns_c[0])
            tail = ns_c >= n_start
            C = (
                fit_const(ns_c[tail], cn_up[tail], m, d)
                if tail.sum() >= 2
                else fit_const(ns_c, cn_up, m, d)
            )
            n_line = np.geomspace(max(n_start, 2.0), gn[-1], 60)

            # Spectral tail from the orbit formula (see gelfand_tails): a rigorous lower
            # reference at every n and exact when n closes a complete Fourier shell.  At a
            # mid-shell index, a real invariant subspace cannot retain only part of a multiplet.
            ex = np.sqrt(
                np.clip(
                    PeriodicSobolevMixedKernel(m=m, d=d).gelfand_tails(int(ns_c[-1])),
                    0.0,
                    None,
                )
            )
            print(
                f"      orbit-tail check: max |c_n^-/tail - 1| = "
                f"{np.abs(cn_lo / ex[ns_c] - 1).max():.1e}, "
                f"max c_n^+/tail = {np.max(cn_up / ex[ns_c]):.3f}"
            )

            # --- top: estimated sampling numbers, numerical c_n lower/upper band, asymptotic guide ---
            top.loglog(
                gn[2:],
                g[2:],
                "-",
                color=col,
                lw=1.6,
                label=rf"$g_n^{{\mathrm{{lin}}}}$ (P-greedy estimate), $m={m}$",
            )
            if diagnostic_overlays:
                top.loglog(
                    np.arange(1, len(ex)),
                    ex[1:],
                    "--",
                    color="k",
                    lw=0.9,
                    alpha=0.55,
                    label="orbit spectral tail (exact at closed shells)" if m == min(m_nodes) else None,
                )
            widths.plot_gelfand_bounds(
                top,
                ns_c,
                cn_lo,
                cn_up,
                certified,
                color=col,
                series_label=rf"$m={m}$",
            )
            top.loglog(
                n_line,
                C * bench_rate(n_line, m, d),
                ":",
                color=col,
                lw=1.4,
                label=rf"$n^{{-{m-0.5:g}}}(\log n)^{{{(d-1)*m}}}$ (asympt. guide)",
            )

            print(
                f"  m={m}: n_used={n_used:4d}  median estimated g_n/c_n {ratio_kind}="
                f"[{np.median(ratio_lo):.2f}, {np.median(ratio_hi):.2f}]  "
                f"({int(certified.sum())}/{len(certified)} uppers certified)"
            )

        top.set_xlabel(r"$n$  (sample nodes / width index)")
        top.set_ylabel(r"$L_\infty$ width")
        top.set_title(
            rf"$H^m_{{\mathrm{{mix}}}}([0,1]^{d})$: estimated sampling $g_n^{{lin}}$ vs "
            rf"Gelfand lower/upper values"
        )
        top.legend(fontsize=7.5)
        top.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        r"Periodic mixed Sobolev: estimated sampling numbers and Gelfand lower/upper values"
    )
    fig.tight_layout()
    fig.savefig("periodic_mixed.png", dpi=130, bbox_inches="tight", pad_inches=0.02)
    print("\nfigure saved -> periodic_mixed.png")


# --------------------------------------------------------------------------- #
#  (3)  the greedy points in [0,1]^2
# --------------------------------------------------------------------------- #
def points_figure(m: int = 2, counts=(64, 256, 576), sel_grid: int = 24000):
    _, ctrs, n_used = greedy_curve(m, d=2, n_nodes=max(counts), sel_grid=sel_grid)
    counts = [c for c in counts if c <= n_used]
    fig, axes = plt.subplots(1, len(counts), figsize=(4.2 * len(counts), 4.4))
    for ax, N in zip(np.atleast_1d(axes), counts):
        P = ctrs[:N]
        ax.scatter(P[:, 0], P[:, 1], s=9, color="k")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"first {N} greedy points")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    fig.suptitle(
        rf"P-greedy design in $[0,1]^2$ for $H^{m}_{{\mathrm{{mix}}}}$ "
        rf"(periodic mixed Sobolev, $m={m}$)"
    )
    fig.tight_layout()
    fig.savefig(
        "periodic_mixed_points.png", dpi=130, bbox_inches="tight", pad_inches=0.02
    )
    print("figure saved -> periodic_mixed_points.png")


if __name__ == "__main__":
    rates_figure()
    points_figure()

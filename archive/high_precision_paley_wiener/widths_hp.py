"""
High-precision (mpmath) Gelfand-width bracket for band-limited kernels -- the arbitrary-precision
twin of widths.gelfand_widths, ARCHIVED (see README.md for why it was retired from the live pipeline).

Recovers c_n past the float64 cliff where a band-limited Gram floors at ~sqrt(eps)*sqrt(K(x,x)).
Depends only on numpy + mpmath and a kernel exposing kfun_mp + prolate_spectrum_hp (see kernels_hp.py).
"""

from __future__ import annotations
import numpy as np
from mpmath import mp, mpf, matrix, sqrt as msqrt


def _golden_max(f, a, b, iters: int = 44):
    """maximize f on [a,b] by golden section (derivative-free, monotone bracket shrink -> the
    estimate only moves UP toward the true local max, i.e. the safe direction for an upper bound).
    """
    gr = (msqrt(5) - 1) / 2
    cN = b - gr * (b - a)
    dN = a + gr * (b - a)
    fc, fd = f(cN), f(dN)
    for _ in range(iters):
        if fc > fd:
            b, dN, fd = dN, cN, fc
            cN = b - gr * (b - a)
            fc = f(cN)
        else:
            a, cN, fc = cN, dN, fd
            dN = a + gr * (b - a)
            fd = f(dN)
    return fc if fc > fd else fd


def gelfand_widths_hp(
    kernel,
    n_list,
    M: int | None = None,
    dps: int = 50,
    n_iter: int = 12,
    n_scan: int = 400,
    ncand: int = 24,
    floor=mpf("1e-3"),
    verbose: bool = False,
):
    r"""High-precision (mpmath) Gelfand-width bracket [c_lo, c_hi] of a band-limited `kernel` on
    [-1,1] for each n in n_list -- the arbitrary-precision twin of gelfand_widths, reaching past the
    float64 cliff where the band-limited Gram floors at ~sqrt(eps)*sqrt(K(x,x)).

    Runs the same IRLS minimax as gelfand_widths, but on the EXACT prolate mode coordinates
        c(x)_k = sqrt(lambda_k) phi_k(x),   ||c(x)||^2 = sum_k lambda_k phi_k(x)^2 = K(x,x),
    with (lambda_k, phi_k) the Slepian spectrum from kernel.prolate_spectrum_hp.  Per n:
        c_lo(n) = sqrt( max_p sum_{k>n} lambda_k(C_p) )      rigorous (weak duality),
        c_hi(n) = sqrt( sup_x [ K(x,x) - ||P_V c(x)||^2 ] )  certified (scan + golden section + x=+-1).
    Everything is in tail form (positive mode sums, no cancellation), so the bracket stays tight
    (ratio ~1.3) many orders below the float64 floor; where both are reliable it agrees with
    gelfand_widths to a few percent (independent cross-check).

    The kernel must supply kfun_mp (arbitrary-precision K) and prolate_spectrum_hp.
    M     -- Gauss-Legendre nodes for the prolate solve (auto ~ max(2 N_eff, 1.5 max(n_list))+12,
             capped at 190 so mpmath eigsy stays in memory/time; >~ 3 N_eff resolves the tail).
    dps   -- mpmath working precision (default 50; the deepest lambda ~ c_hi^2 must stay above
             10^-dps -- 50 digits covers every bandwidth here, deepest ~1e-31 at N_eff=40).
    Returns a dict of float64 arrays: n, c_lo, c_hi (the bracket), sigma (=sqrt(lam_n)), argfrac
    (1-|x*| of the certified sup), plus M, dps, N_eff.  c_lo is rigorous; c_hi a certified sup.
    """
    n_list = [int(v) for v in n_list]
    n_eff = kernel.n_eff
    n_max = max(n_list)
    if M is None:
        # ~2 N_eff resolves the flat prolate band (Nyquist of the bandlimited kernel); ~1.5 n_max
        # converges the tail down to the deepest requested mode.  Capped so the pure-Python mpmath
        # eigsy stays well-conditioned in memory/time (M~2.8 N_eff already OOMs at N_eff=80).
        M = min(max(int(2.0 * n_eff), int(1.5 * n_max)) + 12, 190)
    mp.dps = dps
    K = kernel.kfun_mp
    diag = K(mpf(0))  # K(x,x) = c/pi

    lam, t, w, V = kernel.prolate_spectrum_hp(M, dps)
    sw = [msqrt(wi) for wi in w]
    rlam = [msqrt(lam[k]) for k in range(M)]
    KMAX = min(M, n_max + 16)  # modes beyond this are negligible in every bracket

    def cvec(x, Kw):
        """c(x)_k = sqrt(lam_k) phi_k(x) = (1/sqrt(lam_k)) sum_j sqrt(w_j) K(x,t_j) v_{jk}."""
        kv = [sw[j] * K(x - t[j]) for j in range(M)]
        out = []
        for k in range(Kw):
            s = mpf(0)
            vk = V[k]
            for j in range(M):
                s += kv[j] * vk[j]
            out.append(s / rlam[k])
        return out

    # scan points: dense interior (resolve equioscillation) + geometric endpoint clusters
    xs = [mpf(str(v)) for v in np.linspace(-0.9995, 0.9995, n_scan)]
    for sgn in (mpf(1), mpf(-1)):
        xs.append(sgn)
        for a in np.linspace(0.5, 11.0, 80):
            xs.append(sgn * (1 - mpf(10) ** (-mpf(str(a)))))
    Nx = len(xs)
    Cfull = [cvec(x, KMAX) for x in xs]  # Nx x KMAX mode coordinates (hp), reused per n

    def irls(n):
        """optimal n-dim subspace (KMAX x n padded) + rigorous measure lower bound."""
        Kw = min(n + 14, KMAX)
        C = [row[:Kw] for row in Cfull]
        p = [mpf(1) / Nx] * Nx
        best_up, best_U, best_lo = mpf("inf"), None, mpf(0)
        for _ in range(n_iter):
            rp = [msqrt(pi_) for pi_ in p]
            G = matrix(Kw, Kw)
            for ix in range(Nx):
                ci = C[ix]
                rr = rp[ix]
                sq = [rr * ci[k] for k in range(Kw)]
                for a in range(Kw):
                    sa = sq[a]
                    for b in range(a, Kw):
                        G[a, b] += sa * sq[b]
            for a in range(Kw):
                for b in range(a + 1, Kw):
                    G[b, a] = G[a, b]
            ev, U = mp.eigsy(G)
            Un = [[U[r, Kw - 1 - k] for k in range(n)] for r in range(Kw)]
            r2 = []
            for ix in range(Nx):
                ci = C[ix]
                ss = mpf(0)
                for k in range(n):
                    pr = mpf(0)
                    for r in range(Kw):
                        pr += ci[r] * Un[r][k]
                    ss += pr * pr
                r2.append(diag - ss)
            up = max(r2)
            if up < best_up:
                best_up = up
                best_U = [
                    [Un[r][k] if r < Kw else mpf(0) for k in range(n)]
                    for r in range(KMAX)
                ]
            lo = sum(ev[i] for i in range(Kw - n))
            if lo > best_lo:
                best_lo = lo
            p = [pi_ * (r2[ix] + floor * up) for ix, pi_ in enumerate(p)]
            tot = sum(p)
            p = [pi_ / tot for pi_ in p]
        return best_lo, best_U

    def certified_sup(U, n):
        """sup_x [ c/pi - ||P_V c(x)||^2 ] over the CONTINUOUS box: scan brackets every peak (reusing
        the cached Cfull coordinates), golden-section refines the top candidates, endpoints exact.
        """

        def resid(cx):  # diag - ||P_V cx||^2 for a KMAX-length coordinate vector cx
            ss = mpf(0)
            for k in range(n):
                pr = mpf(0)
                for r in range(KMAX):
                    pr += cx[r] * U[r][k]
                ss += pr * pr
            return diag - ss

        def f(x):  # fresh cvec, only for the off-grid golden-section probes
            return resid(cvec(x, KMAX))

        order = sorted(range(Nx), key=lambda i: xs[i])
        X = [xs[i] for i in order]
        R = [
            resid(Cfull[i]) for i in order
        ]  # scan residuals reuse the cached coordinates
        # endpoints x=+-1 are the extremes of the sorted scan set (both are in xs)
        best, argx = max((R[0], X[0]), (R[-1], X[-1]))
        loc = [p for p in range(1, Nx - 1) if R[p] >= R[p - 1] and R[p] >= R[p + 1]]
        loc.sort(key=lambda p: R[p], reverse=True)
        for p in loc[:ncand]:
            for cand in (R[p], _golden_max(f, X[p - 1], X[p + 1])):
                if cand > best:
                    best, argx = cand, X[p]
        return best, argx

    out = {"n": [], "c_lo": [], "c_hi": [], "sigma": [], "argfrac": []}
    for n in n_list:
        lo, U = irls(n)
        up, argx = certified_sup(U, n)
        clo = msqrt(lo) if lo > 0 else mpf(0)
        chi = msqrt(up) if up > 0 else mpf(0)
        out["n"].append(n)
        out["c_lo"].append(float(clo))
        out["c_hi"].append(float(chi))
        out["sigma"].append(float(rlam[n - 1]) if n - 1 < M else 0.0)
        out["argfrac"].append(float(1 - abs(argx)))
        if verbose:
            print(
                f"  n={n:3d}  c in [{float(clo):.4e}, {float(chi):.4e}]  ratio {float(chi / clo):.3f}",
                flush=True,
            )
    for k in ("n", "c_lo", "c_hi", "sigma", "argfrac"):
        out[k] = np.array(out[k])
    out["M"], out["dps"], out["N_eff"] = M, dps, n_eff
    return out

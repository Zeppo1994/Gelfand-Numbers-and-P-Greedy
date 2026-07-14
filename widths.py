"""
Sampling numbers and Gelfand widths of a bounded-kernel RKHS on the box D=[-1,1]^d, in the
notation of MAIN_Sampl_vs_Gelfand.tex.

  * g_m^lin = inf_{|P|=m} ||Pow_P||_inf -- LINEAR SAMPLING NUMBERS, realized by P-greedy points
    (weak greedy, gamma=1/2): the running sup power of the P-greedy design on a fine grid.
  * c_n = d_n(K)_H = inf_{dim V=n} sup_x dist(a_x, V) -- GELFAND WIDTHS, the Kolmogorov width of
    the translates {K(.,x)} by a reweighted-SVD (IRLS) minimax with a Remez EXCHANGE step,
    returned as a bracket [lower, upper]: lower rigorous (weak duality), upper a branch-and-bound
    CERTIFICATE where the kernel gives a dist_bound modulus (1D stationary) and an off-grid sup
    estimate otherwise (Legendre endpoints, d>1) -- see gelfand_widths.  NOT the top-n Mercer
    eigenspace, which overshoots by ~sqrt(n).
(sigma_n = sqrt(mu_n), exact for a Mercer kernel, are read off kernel.sqrt_mu in legendre.py.)

Theorem under test (Carl-type comparison + Corollary): c_n <= g_m^lin <~ c_n, so g_m/c_n stays
BOUNDED -- the sqrt(n) of the general KPUU bound g_{2n} <= C sqrt(n) c_n drops for a bounded kernel.

Kernel-agnostic: needs kernel.eval / diagonal.  The upper bound sharpens with what the kernel
offers -- dist_bound (rigorous modulus) -> the 1D certificate, eval_grad -> exact d>1 multistart
gradients, feature_map -> cheap Mercer residuals.  dtype follows the kernel (float64; float32
truncates the widths).
"""

from __future__ import annotations
import warnings
import numpy as np
import torch
from scipy.optimize import minimize

from greedy import PGreedy


def box_grid(
    n: int,
    d: int,
    dtype: torch.dtype,
    device="cpu",
    kind: str = "chebyshev",
    domain=(-1.0, 1.0),
    edge_ladder: int = 0,
) -> torch.Tensor:
    """n candidate points in `domain`^d, shape (n, d); with edge_ladder>0 (1D) also a geometric
    endpoint ladder of 2*edge_ladder points.  The measure is kernel-specific (`kind`):
      "chebyshev" -- 1D cos-spaced, dense at +-1, for a BOUNDARY-CONCENTRATED Mercer kernel
          (Legendre: the Christoffel/power function peaks at +-1).  d>1 falls back to Sobol.
      "uniform"   -- scrambled Sobol, the Lebesgue measure for a STATIONARY kernel (Matern, sinc,
          periodic); its even coverage also serves the lower bound's max_p and resolves d>1
          subspaces with far fewer points than i.i.d. uniform.
    edge_ladder -- for a boundary-concentrated WIDTH grid: `edge_ladder` log-spaced offsets per
        endpoint over [1e-10, 1e-1]*(hi-lo)/2.  The width-optimal residual spike sits within ~1/n^2
        of +-1, inside the innermost Chebyshev node (~(pi/n)^2/2); the ladder lets the dual place
        mass there (lifting the lower bound) and the minimax kill the spike (dropping the sup) --
        Legendre s=2, N=1000: n=200 ratio 2.10 -> 1.06 at unchanged cost.  Greedy SELECTION grids
        don't need it (g_m is a sup over placed points), so sampling_vs_gelfand ladders the c_n grid
        only.  (Callers pass the kernel's grid_kind / domain; default stays Chebyshev.)
    """
    lo, hi = domain
    if kind == "chebyshev" and d == 1:
        t = 0.5 * (
            1.0 - torch.cos(torch.linspace(0, np.pi, n, dtype=dtype, device=device))
        )  # in [0,1]
        X = (lo + (hi - lo) * t).reshape(-1, 1)
    else:
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=0)
        X = (lo + (hi - lo) * eng.draw(n).to(dtype)).to(device)
    if edge_ladder and d == 1:
        off = (0.5 * (hi - lo)) * torch.logspace(
            -10, -1.0, int(edge_ladder), dtype=dtype, device=device
        )
        X = torch.cat([X, torch.cat([lo + off, hi - off]).reshape(-1, 1)], 0)
    return X


def log_spaced_ints(hi: int, count: int) -> np.ndarray:
    """`count` distinct ints ~evenly log-spaced over [1, hi] -- the width-reporting abscissae, so
    loglog markers don't pile up at large index.  hi clamped to >=1 (a degenerate n_used==1 -> hi==0
    would hit geomspace's zero-endpoint error)."""
    return np.unique(np.round(np.geomspace(1, max(int(hi), 1), count)).astype(int))


def _refine_sup(
    res2,
    starts,
    dt,
    n_iter: int = 80,
    n_rand=None,
    domain=(-1.0, 1.0),
    res2_vg=None,
    chunk: int = 1024,
):
    """Off-grid ESTIMATE of max_x res2(x), res2 = dist_H(a_x, V_p)^2, over `domain`^d -- an
    estimate, NOT a certificate: an under-converged search can undershoot the true sup (itself a
    valid c_n upper bound, since c_n <= sup over any fixed V_p).  L-BFGS-B multistart from two
    independent polished seed sets -- worst grid nodes and a scrambled-Sobol random set (peaks
    BETWEEN nodes) -- so a missed peak needs both to fail.  `domain` must be the kernel's own: a
    [-1,1] box on a periodic kernel ([0,1] cell) seeds 1-2^{-d} of the random points on periodic
    images (7/8 in d=3).  res2_vg supplies the EXACT gradient (from eval_grad) in one pass; None
    -> central differences (2d evals).  Chunks of `chunk` bound the (chunk,N,d) temporaries.
    Returns (best, v_grid, v_rand, pts, vals): overall max, each seed set's max (v_rand > v_grid =
    worst-node seeds under-covered -- the self-check), and the polished points + res2 values (the
    peaks the exchange feeds back into the dual set)."""
    lo, hi = domain
    grid = starts.reshape(-1, starts.shape[-1])
    K0, d = grid.shape
    dev = grid.device  # run the residual evals on the grid's device (CPU or CUDA)
    n_rand = K0 if n_rand is None else int(n_rand)
    if n_rand:  # random seeds over `domain`^d, polished alongside the grid seeds
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=1)
        rand = (lo + (hi - lo) * eng.draw(n_rand).to(dt)).to(dev)
        z_all = torch.cat([grid.to(dt), rand], 0)
    else:
        z_all = grid.to(dt)
    K = z_all.shape[0]

    def forward(zmat):  # (K,d) -> (K,), chunked forward only, clamped to the box
        zc = np.clip(zmat, lo, hi)
        out = np.empty(zc.shape[0])
        for s in range(0, zc.shape[0], chunk):
            with torch.no_grad():
                zb = torch.as_tensor(zc[s : s + chunk], dtype=dt, device=dev)
                out[s : s + chunk] = res2(zb).cpu().numpy()
        return out

    if (
        res2_vg is not None
    ):  # exact analytic gradient from the kernel's eval_grad (one pass)

        def objective(zf):
            zc = np.clip(zf.reshape(K, d), lo, hi)
            val, g = 0.0, np.empty((K, d))
            for s in range(0, K, chunk):
                with torch.no_grad():
                    v, gg = res2_vg(
                        torch.as_tensor(zc[s : s + chunk], dtype=dt, device=dev)
                    )
                val += float(v.sum())
                g[s : s + chunk] = gg.double().cpu().numpy()
            return -val, -g.reshape(-1)  # minimise the negative sum

        res = minimize(
            objective,
            z_all.cpu().numpy().reshape(-1),
            jac=True,
            method="L-BFGS-B",
            bounds=[(lo, hi)] * (K * d),
            options=dict(maxiter=n_iter),
        )
    else:  # finite-difference fallback

        def f(zf):
            return -float(
                forward(zf.reshape(K, d)).sum()
            )  # seeds independent -> optimise the sum

        def jac(zf):
            z = zf.reshape(K, d)
            h = 1e-6
            g = np.zeros((K, d))
            for c in range(
                d
            ):  # central difference, one batched eval per +/- perturbation
                zp = z.copy()
                zp[:, c] += h
                zm = z.copy()
                zm[:, c] -= h
                g[:, c] = (forward(zp) - forward(zm)) / (2 * h)
            return -g.reshape(-1)

        res = minimize(
            f,
            z_all.cpu().numpy().reshape(-1),
            jac=jac,
            method="L-BFGS-B",
            bounds=[(lo, hi)] * (K * d),
            options=dict(maxiter=n_iter),
        )
    pts = np.clip(res.x.reshape(K, d), lo, hi)
    vals = forward(pts)
    v_grid = float(vals[:K0].max())
    v_rand = float(vals[K0:].max()) if n_rand else 0.0
    return (
        max(v_grid, v_rand),
        v_grid,
        v_rand,
        torch.as_tensor(pts, dtype=dt),
        torch.as_tensor(vals, dtype=dt),
    )


def _bnb_sup(
    res2,
    dist_bound,
    n: int,
    dt,
    domain=(-1.0, 1.0),
    tol: float = 0.02,
    max_evals: int = 400_000,
    chunk: int = 8192,
):
    """CERTIFIED sup_x res2(x) on the 1D box by branch-and-bound, res2 = dist_H(a_x, V_p)^2.  On a
    cell of halfwidth h at center c,  sup r <= r(c) + dist_bound(h)  (r = sqrt(res2)), since
    |r(x)-r(y)| <= ||a_x-a_y||_H <= dist_bound(|x-y|) -- the kernel's rigorous modulus.  Cells whose
    bound exceeds best*(1+tol) are bisected, the rest pruned; the max active-leaf bound is a TRUE
    upper bound at every stage (no seed luck -- this has caught the L-BFGS multistart undershooting
    a true sup by 10%), and cheaper, refining only near the ~n peaks.

    Returns (cert, seen, pts, vals, converged): cert >= sup res2 (rigorous always, loose if the
    budget ran out), seen = largest evaluated res2 (<= sup, doubles as the estimate), (pts, vals) =
    peak seeds for the exchange -- the level-0 scan's local maxima (~one per peak; deeper levels
    cluster on the global peak alone) plus all deeper evals -- and converged = cert within ~3*tol of
    seen.  It can't converge where the widths hit the float64 noise floor (band-limited kernel past
    its cliff, r ~ 1e-7); the caller then falls back to `seen`."""
    lo, hi = domain

    def rvals(c):  # r = sqrt(res2) at the cell centers, chunked
        out = torch.empty(c.shape[0], dtype=dt)
        for s in range(0, c.shape[0], chunk):
            with torch.no_grad():
                out[s : s + chunk] = res2(c[s : s + chunk].reshape(-1, 1))
        return torch.sqrt(torch.clamp(out, min=0.0))

    M0 = min(max(8 * n, 256), max_evals)
    edges = torch.linspace(lo, hi, M0 + 1, dtype=dt)
    c = 0.5 * (edges[:-1] + edges[1:])
    h = torch.full((M0,), (hi - lo) / (2 * M0), dtype=dt)
    r = rvals(c)
    all_c, all_r = [c], [r]
    n_evals = M0
    best = float(r.max())
    cert = float((r + dist_bound(h)).max())
    while True:
        bound = r + dist_bound(h)
        keep = bound > best * (1.0 + tol)
        if not bool(keep.any()):
            cert = best * (1.0 + tol)  # every leaf bound <= threshold: converged
            break
        cert = float(
            bound[keep].max()
        )  # rigorous if we stop here (max over active leaves)
        if n_evals + 2 * int(keep.sum()) > max_evals:
            break  # budget exhausted: cert stays rigorous, just loose (converged=False)
        hn = h[keep] / 2
        c = torch.cat([c[keep] - hn, c[keep] + hn])
        h = torch.cat([hn, hn])
        r = rvals(c)
        n_evals += c.shape[0]
        best = max(best, float(r.max()))
        all_c.append(c)
        all_r.append(r)
    r0 = all_r[
        0
    ]  # peak seeds: level-0 local maxima (~one per peak) + deeper evaluations
    im = torch.zeros(M0, dtype=torch.bool)
    im[1:-1] = (r0[1:-1] >= r0[:-2]) & (r0[1:-1] >= r0[2:])
    im[0], im[-1] = r0[0] >= r0[1], r0[-1] >= r0[-2]
    pts = torch.cat([all_c[0][im]] + all_c[1:]).reshape(-1, 1)
    vals = torch.cat([all_r[0][im]] + all_r[1:]) ** 2
    return cert**2, best**2, pts, vals, cert <= (1.0 + 3.0 * tol) * max(best, 1e-300)


def _pick_peaks(pts, vals, n_keep: int, min_sep: float, extra=None):
    """Dedup peak candidates to <= n_keep exchange points: greedy in descending res2 under a
    pairwise min separation (the multistart / B&B pile many evals on one peak).  `extra` rows
    (e.g. the endpoint-sweep argmax) are appended unconditionally.  pts (M, d), vals (M,)."""
    order = torch.argsort(vals, descending=True)
    if order.shape[0] > 40 * n_keep:  # pre-thin: the tail can't win a slot
        order = order[: 40 * n_keep]
    chosen = []
    for i in order.tolist():
        x = pts[i]
        if all(float(torch.linalg.vector_norm(x - y)) >= min_sep for y in chosen):
            chosen.append(x)
            if len(chosen) >= n_keep:
                break
    if extra is not None:
        chosen.extend(list(extra))
    return torch.stack(chosen) if chosen else pts[:0]


def _extend_system(kernel, Xg, lam, U, Phi_r, nrm2, Xp):
    """EXACT rank-k extension of the truncated working system by k peak points Xp -- a Nystrom
    basis update, no re-eigendecomposition, O(N r_n k).  The span grows from W = top r_n grid modes
    to W' = W + span{(I-Pi_W) a_p}, in which the peak translates lie EXACTLY, so stage 2 can kill
    their residuals.  With C = Pi_W-coords of a_p, S = K(Xp,Xp) - C C^T the residual Gram (out-of-span
    + below-r_n in-span mass), R = sqrtm(S), Rp = its pseudo-inverse:
        Phi2   = [[Phi_r, Z], [C, R]],  Z = (K(Xg,Xp) - Phi_r C^T) Rp -- exact coords of all N+k
                 translates in the extended basis (row norms <= true, so stage 2's lower bound keeps
                 stage 1's truncation semantics);
        nrm2_2 = true squared norms; C, Rp, isq collapse the stage-2 subspace into an (n, N+k) map."""
    dt = Phi_r.dtype
    r_n = Phi_r.shape[1]
    Kxs = kernel.eval(Xp, Xg).to(dt)  # (k, N)
    Kpp = kernel.eval(Xp, Xp).to(dt)  # (k, k)
    lam_r = lam[-r_n:]
    isq = 1.0 / torch.sqrt(lam_r.clamp_min(lam_r.max() * 1e-13))  # (r_n,)
    C = (Kxs @ U[:, -r_n:]) * isq  # (k, r_n) top-mode coordinates of the peaks
    S = Kpp - C @ C.T
    es, Es = torch.linalg.eigh(0.5 * (S + S.T))  # symmetrized residual Gram
    tau = float(es.max().clamp_min(0)) * 1e-12
    sq = torch.sqrt(es.clamp_min(0))
    R = (Es * sq) @ Es.T  # S = R R (symmetric PSD square root)
    Rp = (
        Es * torch.where(es > tau, 1.0 / sq.clamp_min(1e-300), torch.zeros_like(es))
    ) @ Es.T
    Z = (Kxs.T - Phi_r @ C.T) @ Rp  # (N, k) grid coordinates on the new directions
    Phi2 = torch.cat(
        [torch.cat([Phi_r, Z], 1), torch.cat([C, R], 1)], 0
    )  # (N+k, r_n+k)
    nrm2_2 = torch.cat([nrm2, torch.diagonal(Kpp)])
    return Phi2, nrm2_2, C, Rp, isq


def gelfand_widths(
    kernel,
    X,
    n_list,
    n_iter: int = 20,
    floor: float = 1.0,
    rank=None,
    refine: bool = True,
    refine_starts: int = 64,
    refine_iters: int = 80,
    exchange: bool | None = None,
    exch_iter: int = 10,
    certify_tol: float = 0.02,
    certify_evals: int = 400_000,
):
    """Gelfand widths c_n = d_n(K)_H by a reweighted-SVD (IRLS) minimax with a Remez EXCHANGE step,
    as a monotone bracket (upper, lower).  WHAT IS GUARANTEED:
      LOWER -- RIGOROUS (weak duality / Eckart-Young): c_n >= lower, up to the r_n truncation below
               (measured <= 1e-4 relative).
      UPPER -- a branch-and-bound CERTIFICATE (c_n <= upper to within certify_tol, no seed luck) on
               a 1D grid of a dist_bound kernel (Matern, sinc, periodic); elsewhere (Legendre
               endpoints, d>1) an ESTIMATE via L-BFGS multistart + self-check + a 1D-Mercer endpoint
               sweep, holding only if the sup search resolved.

    Dual ascent is the linear IRLS reweight w <- w*(r2 + floor*r2max).  `floor` is the DAMPING knob,
    not just a zero guard: floor=1 (default) scales weights by r2/r2max + 1 in [1, 2], a smooth step
    whose stable trajectory makes the grid-max subspace selection well-behaved OFF-grid; the
    aggressive floor=1e-3 climbs the dual slightly faster but oscillates (Matern 1D ratios 1.19-1.28
    vs 1.04-1.08; Legendre 1.29 vs 1.06 at n=200, with no better lower bound).  One eigendecomposition
    gives Phi = U sqrt(Lam); per n only the top r_n = 3n+100 modes (where V_p lives) enter, matching
    full rank to ~1e-4 -- the only lower-bound caveat, as the compressed subspace can slightly
    overstate the dual tail (<= 1e-4 rel here).  Each IRLS step takes the top-n right-singular
    subspace V_p and brackets  sum_i p_i dist(a_x_i, V_p)^2 <= c_n^2 <= sup_x dist(a_x, V_p)^2.

    EXCHANGE (default on without a feature_map; `exchange` overrides): stage-1's off-grid residual
    peaks rejoin the dual point set, the basis is extended EXACTLY (_extend_system, O(N r_n k)), IRLS
    re-runs warm for exch_iter steps, and the re-optimized sup is taken -- the bracket keeps max of
    lowers, min of uppers (each valid alone).  This attacks the dominant gap, a grid-optimal subspace
    that is bad OFF-grid: Matern nu=3/2 (1D, N=1000) n=200=N/5 ratio 1.69 -> ~1.05 at ~1.7x cost,
    beating doubling n_iter (1.20) or the grid (1.63); d=3 (N=3000) 1.83 -> 1.56.  Auto-OFF for a
    boundary-concentrated Mercer kernel (the endpoint spike just re-emerges at a new offset); there
    the box_grid edge_ladder fixes the cause and the endpoint sweep resolves the sup.

    refine=False returns the grid max (no certificate, no exchange).  rank pins r_n (=N: full).
    Where the certificate can't converge within certify_evals -- widths at the float64 floor, e.g.
    the band-limited kernel past N_eff -- it falls back to the B&B's best evaluated value and warns.
    Reliable for n <~ N/5; beyond it V_p interpolates the N translates, the residual vanishes at
    every node, and the upper bound collapses (sampling_vs_gelfand caps n there)."""
    dt = kernel.dtype
    Xg = X.reshape(X.shape[0], -1).to(dt)
    G = kernel.eval(Xg, Xg).to(dt)
    N = G.shape[0]
    lam, U = torch.linalg.eigh(G)  # ascending; ONE decomposition, shared across all n
    Phi_all = U * torch.sqrt(
        lam.clamp_min(0)
    )  # (N, N) full basis; top modes are last cols
    nrm2 = torch.diag(G).clone()  # true ||a_x||^2 = K(x,x)
    # make_res2(A, Xp) -> (res2, res2_vg): res2(x) = dist(a_x, V_p)^2 for the off-grid sup; A maps
    # the kernel columns (K(x, p))_{p in Xp} to V_p coordinates (Xp = grid for stage 1, grid+peaks
    # for the exchange).  res2_vg gives the exact eval_grad gradient, or None -> finite difference.
    # Mercer kernels work in feature space (M = A Phi_pts, cached for the grid); L-BFGS-B is CPU-only
    # (the heavy LA above runs on Xg.device).
    Xg_cpu = Xg.cpu()
    domain = getattr(kernel, "domain", (-1.0, 1.0))  # box the sup search runs over
    if hasattr(kernel, "feature_map"):
        Phi_grid = kernel.feature_map(Xg_cpu)  # (N, n_trunc), computed ONCE (CPU)

        def make_res2(A, Xp):
            Phi_pts = Phi_grid if Xp is Xg_cpu else kernel.feature_map(Xp)
            M = A @ Phi_pts  # (n, n_trunc): V_p in feature space

            def res2(xq):
                F = kernel.feature_map(xq)
                return torch.clamp(
                    torch.sum(F * F, 1) - torch.sum((F @ M.T) ** 2, 1), min=0.0
                )

            # Legendre feature-map gradient ~1/(1-x^2) is singular at +-1 -> use FD (res2_vg=None)
            return res2, None

    else:  # stationary kernel: cheap closed-form eval, K(x,x) constant
        eval_grad = getattr(kernel, "eval_grad", None)  # analytic dK/dx if provided

        def make_res2(A, Xp):
            def res2(xq):
                b = kernel.eval(xq, Xp) @ A.T
                return torch.clamp(kernel.diagonal(xq) - torch.sum(b * b, 1), min=0.0)

            if eval_grad is None:
                return res2, None

            def res2_vg(xq):  # exact (val, grad) from eval_grad, one pass
                b = kernel.eval(xq, Xp) @ A.T  # (K, n)
                unclamped = kernel.diagonal(xq) - torch.sum(b * b, 1)  # (K,)
                bA = b @ A  # (K, |Xp|); |Xp| = N (stage 1) or N+k (exchange)
                Kg = eval_grad(xq, Xp)  # (K, |Xp|, d) = dK(x, Xp_i)/dx
                # res2 = diag - ||b||^2 (diag const) -> grad = -2 sum_i (bA)_i dK_i/dx, zeroed
                # where res2 clamps to 0 (kept consistent with the value).
                g = -2.0 * torch.einsum("ki,kid->kd", bA, Kg)  # (K, d)
                return (
                    torch.clamp(unclamped, min=0.0),
                    g * (unclamped > 0).to(g.dtype)[:, None],
                )

            return res2, res2_vg

    d_in = Xg.shape[1]
    mercer_1d = hasattr(kernel, "feature_map") and d_in == 1
    certify = refine and d_in == 1 and hasattr(kernel, "dist_bound")
    do_exch = refine and (not mercer_1d if exchange is None else bool(exchange))
    lo_d, hi_d = domain

    def _irls(Phi_w, nrm2_w, w, iters, n):
        """IRLS dual ascent on one working system (stage 1: grid; stage 2: grid+peaks).  Returns
        the best iterate's (grid max, subspace V_p, residuals), the best dual value seen (the
        rigorous lower bound), and the final weights (warm start for the next call)."""
        ub, lb, Vb, r2b = float("inf"), 0.0, None, None
        for _ in range(iters):
            p = w / w.sum()  # probability weights
            sq = torch.sqrt(p)[:, None] * Phi_w  # (N, r) reweighted features
            # top-n right-singular subspace of sq = top-n eigenvectors of the r x r Gram sq^T sq,
            # cheaper than svd(N x r) (condition-squaring touches only the discarded tail).
            _, evecs = torch.linalg.eigh(sq.T @ sq)  # ascending; top-n = last n columns
            Vn = evecs[:, -n:]  # (r, n) top-n right singular vectors; span = V_p
            proj = Phi_w @ Vn
            r2 = torch.clamp(nrm2_w - torch.sum(proj * proj, dim=1), min=0.0)
            r2max = float(r2.max())
            lbi = float((p * r2).sum())  # this iterate's dual value <r, p>
            if r2max < ub:  # keep the best V_p (smallest grid max) for the sup stage
                ub, Vb, r2b = r2max, Vn.T.clone(), r2
            lb = max(lb, lbi)  # weighted-avg residual: lower bound on c_n
            if r2max <= 0.0:
                break  # residual exhausted -> subspace already exact on the working set
            w = w * (r2 + floor * r2max)  # linear IRLS reweight + floor damping
            w = w / w.max()
        return ub, lb, Vb, r2b, w

    def _edge_sweep(res2):
        """Dense log-spaced sweep of both endpoints (1-|x| in [1e-10, ~3e-2]): resolves the
        1D-Mercer boundary spike exactly -- it sits within ~1/N^2 of +-1, unreachable by grid
        nodes or random seeds -- for one feature-map eval, deterministically (no seed luck)."""
        off = torch.logspace(-10, -1.5, 2000, dtype=dt)
        edge = torch.cat([lo_d + off, hi_d - off]).reshape(-1, 1)
        v = res2(edge)
        j = int(torch.argmax(v))
        return float(v[j]), edge[j].reshape(1, -1)

    def _sup_of(A, Xp, r2_grid, ub_grid, n):
        """Upper value for the subspace mapped by A over Xp, plus exchange peak seeds:
        (ub, certified, pts, vals, extra).  1D + dist_bound -> the branch-and-bound CERTIFICATE
        (falling back to its dense-scan best if it can't converge at the float64 floor); else the
        L-BFGS multistart estimate with kernel-aware seeds -- a STATIONARY kernel gets ~3n worst-node
        seeds (~one per inter-node peak, cheap via its analytic gradient), a 1D MERCER kernel keeps
        the multistart as an interior net (refine_starts seeds) with the endpoint sweep authoritative."""
        res2, res2_vg = make_res2(A.cpu(), Xp)
        if certify:
            cert, seen, pts, vals, ok = _bnb_sup(
                res2, kernel.dist_bound, n, dt, domain, certify_tol, certify_evals
            )
            return (cert if ok else max(ub_grid, seen)), ok, pts, vals, None
        n_seed = refine_starts if mercer_1d else max(refine_starts, 3 * n)
        idx = torch.topk(r2_grid, min(n_seed, r2_grid.shape[0])).indices.cpu()
        est, v_grid, v_rand, pts, vals = _refine_sup(
            res2, Xp[idx], dt, refine_iters, domain=domain, res2_vg=res2_vg
        )
        ub = max(ub_grid, est)
        extra = None
        if mercer_1d:
            sweep, arg = _edge_sweep(res2)
            ub = max(ub, sweep)
            extra = arg
        # self-check (one-sided): random seeds out-climbing the worst-node seeds means the latter
        # under-covered.  We keep the max, but a big gap warns the sup is marginally resolved here.
        disagree.append((max(0.0, v_rand - v_grid) / max(v_grid, 1e-30), n))
        return ub, False, pts, vals, extra

    upper, lower, uncert_ns, disagree = [], [], [], []
    # Warm-start the dual weights across the ascending n_list: the worst-case weight moves smoothly
    # in n, so seeding each n from the previous n's final weights cuts the re-convergence iterations.
    w = torch.ones(N, dtype=dt, device=G.device)
    for n in n_list:
        n = int(n)
        # per-n rank: V_p lives in the top ~2n modes, so r_n = 3n+100 matches full rank while
        # keeping the r_n x r_n Gram eigenproblem small.  rank=<int> pins it.
        r_n = min(N, 3 * n + 100) if rank is None else min(N, int(rank))
        Phi = Phi_all[:, -r_n:]  # (N, r_n) top r_n modes
        ub, lb, Vb, r2b, w = _irls(Phi, nrm2, w, n_iter, n)
        if refine and Vb is not None:
            isq = 1.0 / torch.sqrt(lam[-r_n:].clamp_min(lam[-r_n:].max() * 1e-13))
            A1 = (Vb * isq) @ U[:, -r_n:].T  # kernel columns -> V_p coordinates (n, N)
            ub, okn, pts, vals, extra = _sup_of(A1, Xg_cpu, r2b, ub, n)
            if do_exch and pts.shape[0]:
                # EXCHANGE: the found peaks join the dual set; a short warm IRLS re-optimizes the
                # subspace against them and its sup replaces the upper bound when smaller.  Each
                # stage's bound is valid alone -> max of lowers, min of uppers.
                peaks = _pick_peaks(
                    pts,
                    vals,
                    n_keep=min(int(1.5 * n) + 8, 400),  # ~1 slot per residual peak
                    min_sep=(hi_d - lo_d) / (8.0 * max(n, 4)),
                    extra=extra,
                )
                if peaks.shape[0]:
                    peaks = peaks.to(dtype=dt, device=Xg.device)
                    Phi2, nrm2_2, C, Rp, isq2 = _extend_system(
                        kernel, Xg, lam, U, Phi, nrm2, peaks
                    )
                    k = peaks.shape[0]
                    # new points enter at the current max weight: they ARE the worst points known
                    w2 = torch.cat(
                        [w, torch.full((k,), float(w.max()), dtype=dt, device=w.device)]
                    )
                    ubg2, lb2, Vb2, r2b2, _ = _irls(Phi2, nrm2_2, w2, exch_iter, n)
                    lb = max(lb, lb2)
                    if Vb2 is not None:
                        # collapse the stage-2 subspace into a kernel-column map over grid+peaks
                        V_old, V_new = Vb2[:, :r_n], Vb2[:, r_n:]
                        A2 = torch.cat(
                            [
                                ((V_old - V_new @ Rp @ C) * isq2) @ U[:, -r_n:].T,
                                V_new @ Rp,
                            ],
                            1,
                        )
                        Xg2_cpu = torch.cat([Xg_cpu, peaks.cpu()], 0)
                        ub2, ok2, _, _, _ = _sup_of(A2, Xg2_cpu, r2b2, ubg2, n)
                        if ub2 < ub:
                            ub, okn = ub2, ok2
            if certify and not okn:
                uncert_ns.append(n)
        upper.append(ub**0.5)
        lower.append(lb**0.5)
    # Monotone envelope (c_n is non-increasing in n; n_list ascending, as log_spaced_ints builds it):
    #   LOWER  c_n >= max_{m>=n} c-_m -- each c-_m is a rigorous lower bound, so borrowing a larger
    #          one from m>=n is valid.
    #   UPPER  c_n <= min_{k<=n} c+_k -- SUBSPACE NESTING: a resolved sup over the optimal k-dim V
    #          bounds c_n for all n>=k (extend V to n dims; distances only shrink).  A genuine
    #          tightening.  Certified entries stay certificates; estimated ones' falsely-low risk
    #          is guarded by the two-seed-set max (checked via `disagree`).
    lo = torch.flip(
        torch.cummax(torch.flip(torch.tensor(lower, dtype=dt), [0]), dim=0).values, [0]
    )
    up = torch.cummin(torch.tensor(upper, dtype=dt), dim=0).values
    if disagree:
        frac, n_bad = max(disagree)
        if frac > 0.05:
            warnings.warn(
                f"gelfand_widths: sup samplers disagree by {frac:.0%} at "
                f"n={n_bad} -- upper bound may be marginally resolved; raise "
                f"refine_starts/refine_iters or densify the grid.",
                stacklevel=2,
            )
    if uncert_ns:
        warnings.warn(
            f"gelfand_widths: branch-and-bound certificate did not converge within "
            f"certify_evals at n={uncert_ns} (widths at the float64 noise floor?) -- "
            f"the upper bound there is the dense-scan estimate, not a certificate.",
            stacklevel=2,
        )
    return up, lo


def sampling_vs_gelfand(
    kernel,
    d: int = 1,
    m_max: int = 1000,
    sel_grid: int = 20000,
    cn_grid: int = 800,
    n_cn: int = 22,
    n_cap: int | None = None,
    refine_starts: int = 64,
    refine_iters: int = 80,
    edge_ladder: int = 120,
    exchange: bool | None = None,
):
    """Sampling numbers g_m^lin and Gelfand widths c_n of one kernel on [-1,1]^d -- the two curves
    the theorem compares.  dtype follows the kernel.  The c_n bracket machinery (see gelfand_widths)
    is on by default: the exchange step (auto per kernel; `exchange` overrides), the branch-and-bound
    certificate on 1D dist_bound kernels, and -- for a boundary-concentrated (Chebyshev) 1D kernel --
    an edge_ladder appended to the c_n grid only (the greedy selection grid stays ladder-free).

    Returns a dict of log-spaced abscissae and values:
      m_list, g   -- g_m^lin at m in m_list (P-greedy sup power on the selection grid)
      cN, cn      -- c_n at n in cN (cn = upper bracket: B&B certificate on 1D dist_bound kernels,
                     off-grid estimate otherwise; c_n <= cn)
      cn_lo       -- lower bracket, a rigorous lower bound (cn_lo <= c_n <= cn)
      n_used      -- P-greedy centers placed (stops when Pow^2 -> 0)
      d           -- domain dimension.
    refine_starts/refine_iters control the multistart sup search where it still applies (Legendre,
    d>1); the defaults resolve 1D, but a sparser high-d grid needs more starts (see matern.py, d=3)."""
    dt = kernel.dtype
    # Device split (measured): P-greedy is ~14x on GPU (big feature matvecs), but the width
    # IRLS is GPU-hostile in float64 (many small fp64 eighs + syncs).  So greedy on GPU, c_n on CPU.
    gdev = "cuda" if torch.cuda.is_available() else "cpu"
    # candidate measure is kernel-specific: Chebyshev for a boundary-concentrated Mercer kernel,
    # uniform (Sobol) for a stationary one -- the kernel declares it via grid_kind / domain.
    gkind = getattr(kernel, "grid_kind", "chebyshev")
    gdom = getattr(kernel, "domain", (-1.0, 1.0))
    gr = PGreedy(kernel, max_iter=m_max, dtype=dt).fit(
        box_grid(sel_grid, d, dt, gdev, kind=gkind, domain=gdom)
    )
    n_used = gr.n_
    gcurve = (
        gr.g_curve().cpu().numpy()
    )  # g_m over all m (sup power on the selection grid)
    m_list = log_spaced_ints(n_used - 1, 60)  # report g_m at these m (all are cheap)

    # reliable range n <~ N/5 (beyond it V_p interpolates the grid and the upper bound collapses --
    # see gelfand_widths).  Ladder points don't count: they cluster at the endpoints, adding no
    # interior interpolation capacity.
    cap = min(cn_grid - 5, n_used - 1, n_cap or n_used, cn_grid // 5)
    cN = log_spaced_ints(
        cap, n_cn
    )  # compute c_n only here (each is an expensive minimax)
    lad = edge_ladder if (gkind == "chebyshev" and d == 1) else 0
    cn_up, cn_lo = gelfand_widths(
        kernel,
        box_grid(cn_grid, d, dt, "cpu", kind=gkind, domain=gdom, edge_ladder=lad),
        cN,
        refine_starts=refine_starts,
        refine_iters=refine_iters,
        exchange=exchange,
    )
    return dict(
        m_list=m_list,
        g=gcurve[m_list],
        cN=cN,
        cn=cn_up.cpu().numpy(),
        cn_lo=cn_lo.cpu().numpy(),
        n_used=n_used,
        d=d,
    )

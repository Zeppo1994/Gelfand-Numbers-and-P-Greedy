"""
Kernels for the sampling-vs-Gelfand greedy experiments.  Each provides eval, diagonal, and the
PGreedy grid cache (prepare / diag_grid / col); stationary kernels also add an analytic eval_grad
(off-grid width refinement) and a rigorous RKHS modulus dist_bound(h) >= sup ||a_x - a_y||_H over
separations <= h (the certificate widths' 1D branch-and-bound sup needs; Legendre has none -- its
||d/dx a_x||_H diverges at the +-1 endpoints, so no finite modulus exists there).  Definitions and
conventions are in the class docstrings.

  * LegendreMercerKernel -- the Section-7.3 example of Pozharska & Ullrich (arXiv:2103.11124):
    a Mercer kernel with smoothly decaying eigenvalues mu_k = 1/(1 + (k(k+1))^s) and an explicit
    finite feature map (exposed for widths.py).
  * MaternKernel -- half-integer Matern RBF (C^0/C^1/C^2 for nu = 1/2, 3/2, 5/2), stationary,
    any dimension; finite smoothness -> algebraically decaying widths.
  * PeriodicSobolevMixedKernel -- reproducing kernel of the periodic mixed-smoothness Sobolev
    space H^m_mix([0,1]^d); tensor-product mixed-smoothness decay.
  * PaleyWienerSincKernel -- band-limited sinc kernel on [-1,1]; singular numbers flat then
    super-exponential -- a stress test outside the theorem's regularly-varying regime.

Tensors default to float64 (the stability the greedy basis needs); override via `dtype`.
"""

from __future__ import annotations
import heapq
import math
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from scipy.special import bernoulli, comb, factorial


def normalized_legendre(
    x: torch.Tensor, n_trunc: int, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """L2([-1,1], dx)-normalized Legendre polynomials P_0..P_{n_trunc-1}.

    x: (N,) points in [-1,1].  Returns P: (N, n_trunc), P[:, k] = sqrt((2k+1)/2) L_k(x),
    L_k the standard Legendre polynomial (L_k(1)=1).  dtype: recurrence precision (float64).
    """
    x = x.reshape(-1).to(dtype)
    N = x.shape[0]
    L = torch.empty((N, n_trunc), dtype=dtype, device=x.device)
    if n_trunc >= 1:
        L[:, 0] = 1.0
    if n_trunc >= 2:
        L[:, 1] = x
    # Bonnet recurrence: (k+1) L_{k+1} = (2k+1) x L_k - k L_{k-1}
    for k in range(1, n_trunc - 1):
        L[:, k + 1] = ((2 * k + 1) * x * L[:, k] - k * L[:, k - 1]) / (k + 1)
    # normalization factor sqrt((2k+1)/2)
    k_idx = torch.arange(n_trunc, dtype=dtype, device=x.device)
    norm = torch.sqrt((2.0 * k_idx + 1.0) / 2.0)
    return L.mul_(norm)  # (N, n_trunc); in-place -- a second full-size temporary
    # would double the peak memory of a large feature cache


class LegendreMercerKernel:
    """K_s(x,y) = sum_k mu_k P_k(x) P_k(y), mu_k = 1/(1 + (k(k+1))^s), with P_k the L2([-1,1], dx)-
    normalized Legendre polynomials sqrt((2k+1)/2) L_k.  Explicit finite feature map (truncated at
    n_trunc) phi_k = sqrt(mu_k) P_k: K = <phi(x), phi(y)>, phi(x) the RKHS coordinate of a_x =
    K(.,x) -- exposed because widths.py needs it.

    s: smoothness (paper needs s > 1 for a bounded kernel).  n_trunc: Mercer truncation, chosen
    >> #greedy points; the tail is O(k^{1-2s}), so K(x,x) truncation error is O(n_trunc^{2-2s}),
    maximal at x=+-1 (~1/(2 n_trunc^2) for s=2) -- where the width endpoint spike lives.  Keep it
    well below the smallest tracked c_n^2: the c_n endpoint sweep sees the bias in full, while the
    greedy's endpoint-clustered centers cancel it.

    Conventions (verified vs the paper, Section 7.3): reference measure Lebesgue dx, NOT dx/2 --
    giving sup_x sum_{k<m} P_k(x)^2 = m^2/2 at x=+-1, matching N(m)=(m-1)^2/2 up to index shift;
    Legendre eigenvalue k(k+1), so mu_0 = 1."""

    name = "legendre_mercer"
    domain = (-1.0, 1.0)  # box the off-grid width refinement searches over
    grid_kind = (
        "chebyshev"  # boundary-concentrated Mercer: power/Christoffel fn peaks at +-1
    )
    # No eval_grad: the feature-map derivative carries a 1/(1-x^2) (Legendre derivative
    # recurrence), singular at the +-1 endpoints where the sup concentrates -- so widths'
    # off-grid refine uses finite differences for this kernel rather than an analytic gradient.

    def __init__(
        self,
        s: float = 2.0,
        n_trunc: int = 4000,
        dtype: torch.dtype = torch.float64,
        device="cpu",
    ):
        self.s = float(s)
        self.n_trunc = int(n_trunc)
        self.device = device
        self.dtype = dtype
        k = torch.arange(self.n_trunc, dtype=dtype, device=device)
        t = self.s * torch.log(
            k * (k + 1.0)
        )  # (k(k+1))^s in log-space (avoids overflow)
        self.mu = torch.sigmoid(-t)  # Mercer eigenvalues mu_k
        self.sqrt_mu = torch.sqrt(self.mu)  # singular numbers sigma_k

    def feature_map(self, X: torch.Tensor) -> torch.Tensor:
        """phi(X): (N, n_trunc), rows are the RKHS coordinates of a_x = K(.,x)."""
        X = X.reshape(-1)
        P = normalized_legendre(X, self.n_trunc, self.dtype)  # (N, n_trunc)
        return P.mul_(
            self.sqrt_mu.to(P.device)
        )  # broadcast over columns (in-place: P local)

    def eval(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Gram matrix K(X, Y): (Nx, Ny)."""
        return self.feature_map(X) @ self.feature_map(Y).T

    def diagonal(self, X: torch.Tensor) -> torch.Tensor:
        """K(x, x) for each row of X: (Nx,)."""
        Phi = self.feature_map(X)
        return torch.sum(Phi * Phi, dim=1)

    # ---- grid-cache interface used by the greedy loop (avoids recomputing the
    #      Legendre feature map every iteration; centers are always grid points) ----
    def prepare(self, Xd: torch.Tensor):
        self._Xd = Xd.reshape(-1, 1)
        self._Phi = self.feature_map(self._Xd)  # (N, n_trunc), computed ONCE
        self._diag = torch.sum(self._Phi * self._Phi, dim=1)

    def diag_grid(self) -> torch.Tensor:
        return self._diag

    def col(self, j: int) -> torch.Tensor:
        """K(Xd, Xd[j]) = Phi_grid @ phi(x_j): (N,), a cheap matvec."""
        return self._Phi @ self._Phi[j]


class MaternKernel:
    """Matern kernel with half-integer smoothness nu -- a rougher, algebraically decaying
    alternative to the Legendre kernel.  Closed forms (no Bessel), r = ||x-y||, a = sqrt(2 nu)/ell:
        nu = 1/2:  exp(-a r)                          (C^0, roughest)
        nu = 3/2:  (1 + a r) exp(-a r)                (C^1)
        nu = 5/2:  (1 + a r + (a r)^2/3) exp(-a r)    (C^2)
    Exposes eval / eval_grad / diagonal and the greedy grid cache (prepare / diag_grid / col); any d.
    """

    name = "matern"
    domain = (-1.0, 1.0)  # box the off-grid width refinement searches over
    grid_kind = (
        "uniform"  # stationary: no endpoint preference -> Lebesgue candidate measure
    )

    def __init__(
        self,
        nu: float = 1.5,
        ell: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device="cpu",
    ):
        assert nu in (0.5, 1.5, 2.5), "half-integer nu in {0.5, 1.5, 2.5}"
        self.nu = float(nu)
        self.ell = float(ell)
        self.a = math.sqrt(2.0 * self.nu) / self.ell  # inverse correlation length
        self.dtype = dtype
        self.device = device

    def _corr(self, r: torch.Tensor) -> torch.Tensor:
        """Matern correlation as a function of distance r (=1 at r=0, no cusp for the
        half-integer forms used here, so no guard is needed in the forward eval)."""
        ar = self.a * r
        e = torch.exp(-ar)
        if self.nu == 0.5:
            return e
        if self.nu == 1.5:
            return (1.0 + ar) * e
        return (1.0 + ar + ar * ar / 3.0) * e  # nu = 5/2

    def eval(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Gram matrix K(X, Y): (Nx, Ny).  X, Y are (n, d)."""
        X = X.reshape(X.shape[0], -1).to(self.dtype)
        Y = Y.reshape(Y.shape[0], -1).to(self.dtype)
        return self._corr(torch.cdist(X, Y))

    def eval_grad(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """d/dX K(X_i, Y_j): (Nx, Ny, d).  Analytic: grad = [C'(r)/r] (x-y); for nu=3/2, 5/2 the
        C'(r)/r has a closed form with the r cancelled, so it is finite at r=0 (unlike autograd
        through cdist's 0/0 at coincident points); the nu=1/2 cusp is guarded to 0."""
        X = X.reshape(X.shape[0], -1).to(self.dtype)
        Y = Y.reshape(Y.shape[0], -1).to(self.dtype)
        diff = X[:, None, :] - Y[None, :, :]  # (Nx, Ny, d)
        r = torch.linalg.vector_norm(diff, dim=2)  # (Nx, Ny)
        ar = self.a * r
        e = torch.exp(-ar)
        if self.nu == 1.5:
            fac = -(self.a**2) * e  # C'(r)/r = -a^2 e^{-ar}
        elif self.nu == 2.5:
            fac = (
                -(self.a**2 / 3.0) * (1.0 + ar) * e
            )  # C'(r)/r = -(a^2/3)(1+ar) e^{-ar}
        else:  # nu = 1/2: C'(r)/r = -a e^{-ar}/r, singular at r=0 -> guard to 0 (cusp)
            fac = torch.where(
                r > 1e-12, -self.a * e / r.clamp_min(1e-12), torch.zeros_like(r)
            )
        return fac[:, :, None] * diff  # (Nx, Ny, d)

    def diagonal(self, X: torch.Tensor) -> torch.Tensor:
        """K(x, x) = 1 for every point: (Nx,)."""
        return torch.ones(X.shape[0], dtype=self.dtype, device=X.device)

    def dist_bound(self, h: torch.Tensor) -> torch.Tensor:
        """Rigorous RKHS modulus: sup_{||x-y|| <= h} ||a_x - a_y||_H <= sqrt(2(1 - corr(h))).
        Exact for Matern (any d): ||a_x - a_y||^2 = 2(K(0) - K(r)) and the correlation is
        decreasing in r, so the sup over r <= h sits at r = h.  Elementwise on a tensor of
        separations; consumed by the branch-and-bound sup certificate in widths (d=1).
        """
        return torch.sqrt(torch.clamp(2.0 * (1.0 - self._corr(h)), min=0.0))

    def dist_bound_cell(self, H: torch.Tensor) -> torch.Tensor:
        """Cell modulus for the d>1 branch-and-bound sup certificate: the farthest cell point
        sits at the corner, so dist_bound at the half-diagonal ||H||_2 bounds the cell (exact
        for Matern at any d -- the modulus only sees the Euclidean distance)."""
        return self.dist_bound(torch.linalg.vector_norm(H, dim=1))

    # ---- grid-cache interface used by the strong greedy loop ----
    def prepare(self, Xd: torch.Tensor):
        self._Xd = Xd.reshape(Xd.shape[0], -1).to(self.dtype)
        self._diag = torch.ones(
            self._Xd.shape[0], dtype=self.dtype, device=self._Xd.device
        )

    def diag_grid(self) -> torch.Tensor:
        return self._diag

    def col(self, j: int) -> torch.Tensor:
        """K(Xd, Xd[j]): (N,), one column."""
        r = torch.cdist(self._Xd, self._Xd[j : j + 1])  # (N, 1)
        return self._corr(r).reshape(-1)


class PeriodicSobolevMixedKernel:
    r"""Reproducing kernel of the periodic mixed-smoothness Sobolev space
    H^m_mix([0,1]^d) -- Berlinet & Thomas-Agnan, RKHS book, p.318 (Example 19
    tensorized d times).  It is the d-fold tensor product of the univariate kernel

        k_1(s,t) = 1 + (-1)^{m-1}/(2m)! * B_{2m}(|s-t|),

    with B_{2m} the Bernoulli polynomial of degree 2m; m an integer >= 1.  The
    norm is  ||u||^2 = (int u)^2 + int (u^{(m)})^2  per coordinate, mixed across
    coordinates -- i.e. the m-fold MIXED derivative lives in L2.

    Fourier diagonalization (verified numerically): on the 1D torus
        k_1(s,t) = sum_{k in Z} lambda_k e^{2 pi i k (s-t)},
        lambda_0 = 1,   lambda_k = (2 pi |k|)^{-2m}  (k != 0),
    so the tensor eigenvalues are lambda_{k_1..k_d} = prod_j lambda_{k_j} -- the
    mixed-smoothness decay.  Since B_{2m}(x) = B_{2m}(1-x), the argument |s-t| may
    be taken as the periodic distance min(|s-t|, 1-|s-t|) with no change in value.

    Domain [0,1]^d; stationary and periodic, so K(x,x) = k_1(0)^d is constant.  Exposes eval /
    eval_grad / diagonal and the greedy grid cache (prepare / diag_grid / col); any d, inferred
    from the input points (same object serves 1D or 2D).
    """

    name = "periodic_sobolev_mix"
    domain = (0.0, 1.0)  # fundamental cell; a [-1,1] box would seed periodic images
    grid_kind = (
        "uniform"  # stationary/periodic: uniform candidate measure (matches grid01)
    )

    def __init__(
        self, m: int = 1, d: int = 1, dtype: torch.dtype = torch.float64, device="cpu"
    ):
        self.m = int(m)
        self.d = int(d)  # reference dimension (for the theoretical rate); eval infers d
        self.dtype = dtype
        self.device = device
        N = 2 * self.m
        B = bernoulli(N)  # Bernoulli numbers B_0..B_{2m} (B_1 = -1/2)
        # B_{2m}(x) = sum_{j=0}^{2m} C(2m,j) B_j x^{2m-j}; store descending powers.
        coeffs = [float(comb(N, j, exact=True) * B[j]) for j in range(N + 1)]
        self.coeffs = torch.tensor(coeffs, dtype=dtype, device=device)
        self.pref = float((-1.0) ** (self.m - 1) / factorial(N))  # (-1)^{m-1}/(2m)!
        self._k1_0 = 1.0 + self.pref * float(coeffs[-1])  # k_1(0) = 1 + pref*B_{2m}(0)

    def _bern(self, r: torch.Tensor) -> torch.Tensor:
        """B_{2m}(r) by Horner.  Coeffs follow r's device, so the same kernel evaluates
        on GPU tensors (the heavy width linalg) and on CPU tensors (the scipy refine).
        """
        out = torch.zeros_like(r)
        for c in self.coeffs.to(r.device):
            out = out * r + c
        return out

    def _bern_deriv(self, r: torch.Tensor) -> torch.Tensor:
        """B'_{2m}(r) by Horner over the derivative coefficients (coeffs[j] r^{2m-j} ->
        coeffs[j] (2m-j) r^{2m-j-1}, dropping the constant term); used by eval_grad."""
        out = torch.zeros_like(r)
        Nb = 2 * self.m
        for j, c in enumerate(self.coeffs.to(r.device)):
            if j == Nb:  # constant term -> derivative 0
                break
            out = out * r + c * (Nb - j)
        return out

    def _k1(self, r: torch.Tensor) -> torch.Tensor:
        """Univariate kernel as a function of the (periodic) distance r in [0,1]."""
        return 1.0 + self.pref * self._bern(r)

    def eval(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Gram matrix K(X, Y): (Nx, Ny).  X, Y are (n, d) points in [0,1]^d."""
        X = X.reshape(X.shape[0], -1).to(self.dtype)
        Y = Y.reshape(Y.shape[0], -1).to(self.dtype)
        diff = X[:, None, :] - Y[None, :, :]  # (Nx, Ny, d)
        # periodic (torus) distance r in [0,1/2], correct for ANY real inputs: fold by the
        # nearest integer, so points off [0,1]^d (from the off-grid width refine) stay valid.
        r = torch.abs(diff - torch.round(diff))
        return self._k1(r).prod(dim=2)  # tensor product across coordinates

    def eval_grad(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """d/dX K(X_i, Y_j): (Nx, Ny, d), gradient in the first argument.  Analytic -- the
        periodic-distance fold has a clean a.e. derivative -- so widths' off-grid sup refine
        uses exact gradients instead of finite differences.  Per coordinate,
        d/dx_c K = k1'(r_c) sign(fold_c) * prod_{c'!=c} k1(r_c')."""
        X = X.reshape(X.shape[0], -1).to(self.dtype)
        Y = Y.reshape(Y.shape[0], -1).to(self.dtype)
        diff = X[:, None, :] - Y[None, :, :]  # (Nx, Ny, d)
        fold = diff - torch.round(diff)  # periodic residual in [-1/2, 1/2]
        r = torch.abs(fold)
        k1 = self._k1(r)  # (Nx, Ny, d) per-coordinate kernel
        dk1 = self.pref * self._bern_deriv(r) * torch.sign(fold)  # d k1(r_c)/d x_c
        cofactor = (
            k1.prod(dim=2, keepdim=True) / k1
        )  # prod over the OTHER coords (k1>0)
        return cofactor * dk1  # (Nx, Ny, d)

    def diagonal(self, X: torch.Tensor) -> torch.Tensor:
        """K(x, x) = k_1(0)^d for every point: (Nx,)."""
        d = X.reshape(X.shape[0], -1).shape[1]
        return torch.full(
            (X.shape[0],), self._k1_0**d, dtype=self.dtype, device=X.device
        )

    def dist_bound(self, h: torch.Tensor) -> torch.Tensor:
        """UNIVARIATE rigorous RKHS modulus: sup_{|s-t| <= h} ||a_s - a_t||_H =
        sqrt(2(k_1(0) - k_1(min(h, 1/2)))) -- exact because (-1)^{m-1} B_{2m} is decreasing
        on [0, 1/2] (periodic distance never exceeds 1/2).  Only for the d=1 kernel; the
        branch-and-bound sup certificate in widths consumes it on 1D grids only."""
        return torch.sqrt(
            torch.clamp(2.0 * (self._k1_0 - self._k1(torch.clamp(h, max=0.5))), min=0.0)
        )

    def gelfand_tails(self, n_max: int) -> np.ndarray:
        r"""EXACT Gelfand widths of the translate set on the full torus, by symmetry.  The set
        {a_x : x in T^d} is an orbit of the translation group and the kernel is diagonal in
        characters, so (averaging over Haar measure; Pinkus, n-widths of orbits) the top-n
        character span is an optimal subspace and

            c_n^2 = sum_{k > n} lambda_k,   lambda sorted descending, cos/sin pairs twice.

        Exact whenever n closes a degenerate shell (lambda_{n-1} > lambda_n strictly); at a
        mid-shell n it is the exact LOWER end of the interval [sqrt(tail_n), sqrt(tail_shell)]
        containing c_n (a real invariant subspace needs whole cos/sin multiplets).

        Returns tails[n] = c_n^2 for n = 0..n_max: trace k_1(0)^d minus the exact top-n sum
        (top modes enumerated by a lattice max-heap, no truncated-box error).  float64
        cancellation floors the tails near ~1e-16 * k_1(0)^d, i.e. c_n below ~1e-8 is noise.
        """
        Kdim = n_max + 2  # single-axis modes alone outrank anything with k > n_max
        vals = [1.0] + [(2.0 * math.pi * k) ** (-2 * self.m) for k in range(1, Kdim)]
        mult = [1] + [2] * (Kdim - 1)  # +-k <-> cos/sin pair
        heap = [
            (-1.0, (0,) * self.d)
        ]  # (-product eigenvalue, per-dim distinct-value index)
        seen = {(0,) * self.d}
        lam, count = [], 0
        while heap and count <= n_max:
            negv, idx = heapq.heappop(heap)
            m_k = 1
            for j in idx:
                m_k *= mult[j]
            lam.extend([-negv] * m_k)
            count += m_k
            for c in range(self.d):  # lattice neighbors: one index up per coordinate
                nidx = idx[:c] + (idx[c] + 1,) + idx[c + 1 :]
                if nidx not in seen and nidx[c] < Kdim:
                    seen.add(nidx)
                    heapq.heappush(heap, (negv * vals[nidx[c]] / vals[idx[c]], nidx))
        top = np.cumsum(np.array(lam[: n_max + 1]))
        return np.concatenate([[self._k1_0**self.d], self._k1_0**self.d - top])[
            : n_max + 1
        ]

    # ---- grid-cache interface used by the strong greedy loop ----
    def prepare(self, Xd: torch.Tensor):
        self._Xd = Xd.reshape(Xd.shape[0], -1).to(self.dtype)
        self._diag = self.diagonal(self._Xd)

    def diag_grid(self) -> torch.Tensor:
        return self._diag

    def col(self, j: int) -> torch.Tensor:
        """K(Xd, Xd[j]): (N,), one column."""
        return self.eval(self._Xd, self._Xd[j : j + 1]).reshape(-1)


class PaleyWienerSincKernel:
    r"""Reproducing kernel of the Paley-Wiener space of band-limited functions restricted to
    [-1,1] -- the running example of the Strobl slides (Slepian's prolate setting), a bounded
    STATIONARY kernel whose singular numbers do NOT decay algebraically.

        K_c(s,t) = sin(c (s-t)) / (pi (s-t)) = (c/pi) * sinc(c (s-t)/pi),   s,t in [-1,1],

    reproducing PW_c = { f in L2(R) : supp(hat f) subset [-c,c] } (K(x,x) = c/pi).  Its L2([-1,1])
    operator T_c is diagonalized by the prolate spheroidal functions, with Slepian eigenvalues
    lambda_k ~ 1 for k < N_eff and a SUPER-EXPONENTIAL cliff past the effective dimension

        N_eff = 2c/pi   (Shannon number / time-bandwidth product).

    So sigma_k = sqrt(lambda_k) are flat then plunge at k ~ N_eff -- not the regularly-varying decay
    the theorem in the companion manuscript assumes, hence a stress test: sampling-number estimates and c_n are
    ~flat then fall off the same cliff.  Univariate; provides an analytic eval_grad (exact off-grid
    width refinement) and singular_numbers() (the Nystrom prolate spectrum).  Domain [-1,1], so it
    drops into box_grid / widths / greedy unchanged.

    NUMERICAL FLOOR: band-limited => a Gram over N >> N_eff points has numerical rank ~ N_eff, so the
    float64 widths recover reliably only above the floor ~sqrt(eps)*sqrt(K(x,x)) ~ 1e-7 (n <~ N_eff);
    past the cliff the recovered values ARE the floor.  The true super-exponentially small widths there
    would need an arbitrary-precision (mpmath) reimplementation of the width minimax.
    """

    name = "paley_wiener_sinc"
    domain = (-1.0, 1.0)  # box the off-grid width refinement searches over
    grid_kind = (
        "uniform"  # stationary band-limited: Lebesgue candidate measure, not Chebyshev
    )

    def __init__(
        self,
        c: float | None = None,
        n_eff: float | None = None,
        dtype: torch.dtype = torch.float64,
        device="cpu",
    ):
        assert (c is None) != (n_eff is None), "give exactly one of c, n_eff"
        self.c = float(c) if c is not None else float(n_eff) * math.pi / 2.0
        self.n_eff = 2.0 * self.c / math.pi  # Shannon number = time-bandwidth product
        self.diag_val = self.c / math.pi  # K(x,x), constant (stationary)
        self.dtype = dtype
        self.device = device

    def _kfun(self, u: torch.Tensor) -> torch.Tensor:
        """(c/pi) sinc(c u / pi) = sin(c u)/(pi u); torch.sinc gives the c/pi limit at u=0."""
        return self.diag_val * torch.sinc(self.c * u / math.pi)

    def eval(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Gram matrix K(X, Y): (Nx, Ny)."""
        X = X.reshape(-1, 1).to(self.dtype)
        Y = Y.reshape(1, -1).to(self.dtype)
        return self._kfun(X - Y)

    def eval_grad(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """d/dX K(X_i, Y_j): (Nx, Ny, 1).  Analytic: for f(u)=sin(cu)/(pi u),
        f'(u) = [c u cos(c u) - sin(c u)] / (pi u^2), finite at u=0 (odd, -> 0 via the
        -(c^3/3pi) u Taylor branch) -- so widths' off-grid sup refine uses exact gradients.
        """
        X = X.reshape(-1, 1).to(self.dtype)
        Y = Y.reshape(1, -1).to(self.dtype)
        u = X - Y
        cu = self.c * u
        num = cu * torch.cos(cu) - torch.sin(cu)
        small = u.abs() < 1e-6
        u_safe = torch.where(small, torch.ones_like(u), u)
        g = torch.where(
            small, -(self.c**3) * u / (3.0 * math.pi), num / (math.pi * u_safe * u_safe)
        )
        return g.unsqueeze(-1)  # (Nx, Ny, 1)

    def diagonal(self, X: torch.Tensor) -> torch.Tensor:
        """K(x, x) = c/pi for every point: (Nx,)."""
        n = X.reshape(-1, 1).shape[0]
        return torch.full((n,), self.diag_val, dtype=self.dtype, device=X.device)

    def dist_bound(self, h: torch.Tensor) -> torch.Tensor:
        """Rigorous RKHS modulus: sup_{|s-t| <= h} ||a_s - a_t||_H <= min(L h, 2 sqrt(K(0)))
        with L = sup_x ||d/dx a_x||_H = sqrt(-K''(0)) = sqrt(c^3/(3 pi)).  The exact
        2(K(0)-K(h)) is NOT monotone for the oscillatory sinc, so the Lipschitz form is used
        (same small-h asymptotics, ~L h); the 2 sqrt(K(0)) cap is the triangle inequality.
        Consumed by the branch-and-bound sup certificate in widths (d=1)."""
        L = math.sqrt(self.c**3 / (3.0 * math.pi))
        return torch.clamp(L * h, max=2.0 * math.sqrt(self.diag_val))

    def singular_numbers(self, n: int, n_quad: int | None = None) -> np.ndarray:
        """sigma_k = sqrt(lambda_k(c)), k=1..n: the top-n prolate singular numbers of the
        embedding, from a Gauss-Legendre Nystrom discretization of T_c on [-1,1].  These are
        the flat-then-super-exponential Slepian eigenvalues (independent of the greedy/width
        code); used only for the sigma_n overlay and to mark the N_eff cliff."""
        if n_quad is None:
            n_quad = max(200, int(8 * self.n_eff) + 50)
        t, w = leggauss(n_quad)  # nodes/weights on [-1,1]
        U = t[:, None] - t[None, :]
        Kmat = (self.c / math.pi) * np.sinc(
            self.c * U / math.pi
        )  # np.sinc(0)=1 -> c/pi
        sw = np.sqrt(w)
        A = sw[:, None] * Kmat * sw[None, :]  # symmetric similarity of T_c
        lam = np.clip(np.sort(np.linalg.eigvalsh(A))[::-1], 0.0, None)
        return np.sqrt(lam[:n])

    # ---- grid-cache interface used by the strong greedy loop ----
    def prepare(self, Xd: torch.Tensor):
        self._Xd = Xd.reshape(-1, 1).to(self.dtype)
        self._diag = torch.full(
            (self._Xd.shape[0],),
            self.diag_val,
            dtype=self.dtype,
            device=self._Xd.device,
        )

    def diag_grid(self) -> torch.Tensor:
        return self._diag

    def col(self, j: int) -> torch.Tensor:
        """K(Xd, Xd[j]): (N,), one column."""
        return self._kfun(self._Xd - self._Xd[j : j + 1]).reshape(-1)

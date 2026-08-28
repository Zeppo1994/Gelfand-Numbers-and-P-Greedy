"""
P-greedy center selection for kernel interpolation.

The next center is placed where the power function
    Pow_P(x) = dist_H( K(.,x), span{ K(.,x_i) : x_i in P } )
is maximal over a fixed candidate grid.  This is the strong rule (gamma=1) relative to that
finite grid.  It satisfies the weak-greedy rule with gamma=1/2 on the full dictionary of kernel
translates only if the grid resolves the true power-function supremum within a factor of two;
the computation does not verify that off-grid condition.  PGreedy maintains the residual
power at EVERY grid point via an incremental Newton basis (Pazouki-Schaback): O(N)/step,
O(N m^2) total, returning the full numerical sampling-number curve g_m^lin over m centers.  Device-
agnostic (allocates on the input's device) and dtype-threaded (default float64; float32
is faster on GPU but its power update p <- p - v_n^2 cancels badly as Pow -> 0).
"""

from __future__ import annotations
import torch

class PGreedy:
    """Strong P-greedy: exact argmax of the residual power over the grid."""

    def __init__(
        self,
        kernel,
        max_iter=500,
        tol_p=1e-14,
        dtype: torch.dtype = torch.float64,
        verbose=False,
    ):
        self.kernel = kernel
        self.max_iter = int(max_iter)
        self.tol_p = float(tol_p)
        self.dtype = dtype
        self.verbose = verbose

    def fit(self, Xd: torch.Tensor):
        """Run strong greedy over the fixed candidate grid Xd."""
        Xd = Xd.reshape(Xd.shape[0], -1).to(self.dtype)  # (N, d)
        dev = Xd.device
        N, d = Xd.shape
        m = min(self.max_iter, N)

        self.kernel.prepare(Xd)  # cache grid features ONCE
        p = self.kernel.diag_grid().clone()  # running power^2 over grid, init K(x,x)

        V = torch.zeros((N, m), dtype=self.dtype, device=dev)  # Newton basis on grid
        idx_sel = torch.zeros(m, dtype=torch.long, device=dev)
        ctrs = torch.zeros((m, d), dtype=self.dtype, device=dev)
        pmax = torch.zeros(m + 1, dtype=self.dtype, device=dev)
        pmax[0] = torch.sqrt(torch.clamp(p.max(), min=0))  # pmax[0] = R (0 centers)
        n_used = 0

        for n in range(m):
            # strong greedy: exact argmax of the residual power over the grid.
            j = torch.argmax(p)
            pj = p[j]
            # stop on exhausted residual power (pj<=0 would divide by sqrt(0))
            # or once the tolerance is reached.
            if pj <= 0.0 or pj <= self.tol_p:
                break
            idx_sel[n] = j
            ctrs[n] = Xd[j]
            sqrt_pj = pj**0.5

            # new kernel column against the freshly selected center (from cache)
            a_new = self.kernel.col(j)  # (N,), cheap matvec

            # Newton basis function value at all grid points:
            #   v_n(x) = ( K(x, x_j) - sum_{l<n} v_l(x) v_l(x_j) ) / sqrt(pj)
            if n > 0:
                a_new = a_new - V[:, :n] @ V[j, :n]
            v_n = a_new / sqrt_pj
            V[:, n] = v_n

            # incremental power update
            p = torch.clamp(p - v_n * v_n, min=0.0)
            n_used = n + 1
            pmax[n_used] = torch.sqrt(torch.clamp(p.max(), min=0))

            if self.verbose and (n % 50 == 0):
                print(f"  [greedy] n={n_used:4d}  sup Pow = {float(pmax[n_used]):.3e}")

        self.ctrs_ = ctrs[:n_used]
        self.idx_ = idx_sel[:n_used]
        self.n_ = n_used
        # pmax[n] = sup power function with n centers placed
        self.pmax_ = pmax[: n_used + 1].clone()
        return self

    def g_curve(self) -> torch.Tensor:
        """Numerical sampling numbers g_m^lin as a function of the number of centers m.

        Returns tensor of length n_+1; entry m is the value with m centers."""
        return self.pmax_


def power_function(kernel, P: torch.Tensor, Xq: torch.Tensor) -> torch.Tensor:
    """Pow_P(x) = sqrt( K(x,x) - K(x,P) Kpp^-1 K(P,x) ) at every x in Xq, direct from centers P
    (PGreedy tracks only the running sup; this recomputes the whole power function for plots).
    """
    dt = kernel.dtype
    Kpp = kernel.eval(P, P) + 1e-12 * torch.eye(
        P.shape[0], dtype=dt, device=P.device
    )  # jitter for a stable solve
    Kqp = kernel.eval(Xq, P)
    sol = torch.linalg.solve(Kpp, Kqp.T)
    return torch.sqrt(
        torch.clamp(kernel.diagonal(Xq) - torch.sum(Kqp * sol.T, dim=1), min=0.0)
    )

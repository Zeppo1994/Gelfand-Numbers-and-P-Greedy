"""Strong P-greedy center selection on a finite candidate grid.

The implementation maintains the squared power function at every grid point
through an incremental Newton basis. Each step maximizes that grid residual
exactly, but no off-grid accuracy guarantee is inferred. The returned curve is
therefore a numerical sampling-error estimate, not the optimized sampling number.
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
        point_count, dimension = Xd.shape
        capacity = min(self.max_iter, point_count)

        self.kernel.prepare(Xd)  # cache grid features ONCE
        p = self.kernel.diag_grid().clone()  # running power^2 over grid, init K(x,x)

        V = torch.zeros(
            (point_count, capacity), dtype=self.dtype, device=dev
        )  # Newton basis on grid
        idx_sel = torch.zeros(capacity, dtype=torch.long, device=dev)
        ctrs = torch.zeros((capacity, dimension), dtype=self.dtype, device=dev)
        pmax = torch.zeros(capacity + 1, dtype=self.dtype, device=dev)
        pmax[0] = torch.sqrt(torch.clamp(p.max(), min=0))  # pmax[0] = R (0 centers)
        n_used = 0

        for n in range(capacity):
            # strong greedy: exact argmax of the residual power over the grid.
            j = int(torch.argmax(p))
            pj = p[j]
            # stop on exhausted residual power (pj<=0 would divide by sqrt(0))
            # or once the tolerance is reached.
            if pj <= 0.0 or pj <= self.tol_p:
                break
            idx_sel[n] = j
            ctrs[n] = Xd[j]
            sqrt_pj = torch.sqrt(pj)

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
        """Return the grid power maximum after each number of selected centers.

        The tensor has length ``n_ + 1``; entry ``m`` uses ``m`` centers.
        """
        return self.pmax_


def power_function(
    kernel,
    centers: torch.Tensor,
    query: torch.Tensor,
    *,
    jitter: float = 1e-12,
) -> torch.Tensor:
    """Evaluate the power function directly from a fixed center set."""
    gram = kernel.eval(centers, centers)
    gram = gram + jitter * torch.eye(
        centers.shape[0], dtype=kernel.dtype, device=centers.device
    )
    cross = kernel.eval(query, centers)
    coefficients = torch.linalg.solve(gram, cross.T)
    residual = kernel.diagonal(query) - torch.sum(cross * coefficients.T, dim=1)
    return torch.sqrt(torch.clamp(residual, min=0.0))

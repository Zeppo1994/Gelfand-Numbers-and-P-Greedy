"""Strong P-greedy center selection on a finite candidate grid.

The implementation maintains the squared power function at every grid point
through an incremental Newton basis. Each step maximizes that grid residual
exactly, but no off-grid accuracy guarantee is inferred. The returned curve is
therefore a numerical sampling-error estimate, not the optimized sampling number.
"""

from __future__ import annotations

import torch


class PGreedy:
    """Strong P-greedy: exact argmax of the residual power over the grid.

    ``tol_p`` is a threshold on the squared power: selection stops once the
    grid maximum of the squared power function is at most ``tol_p``.
    ``dtype`` defaults to the kernel's dtype.
    """

    def __init__(
        self,
        kernel,
        max_iter=500,
        tol_p=1e-14,
        dtype: torch.dtype | None = None,
        verbose=False,
    ):
        if tol_p < 0.0:
            raise ValueError("tol_p must be nonnegative")
        self.kernel = kernel
        self.max_iter = int(max_iter)
        self.tol_p = float(tol_p)
        self.dtype = kernel.dtype if dtype is None else dtype
        self.verbose = verbose

    def fit(self, Xd: torch.Tensor):
        """Run strong greedy over the fixed candidate grid Xd."""
        Xd = Xd.reshape(Xd.shape[0], -1).to(self.dtype)  # (N, d)
        dev = Xd.device
        point_count = Xd.shape[0]
        capacity = min(self.max_iter, point_count)

        self.kernel.prepare(Xd)  # cache grid features ONCE
        # running squared power over the grid, initialized to K(x,x)
        p = self.kernel.diag_grid().to(self.dtype).clone()

        V = torch.zeros(
            (point_count, capacity), dtype=self.dtype, device=dev
        )  # Newton basis on grid
        idx_sel = torch.zeros(capacity, dtype=torch.long, device=dev)
        pmax = torch.zeros(capacity + 1, dtype=self.dtype, device=dev)
        pmax[0] = torch.sqrt(p.max())  # pmax[0] = R (0 centers)
        n_used = 0

        for n in range(capacity):
            # strong greedy: exact argmax of the residual power over the grid.
            j = int(torch.argmax(p))
            pj = float(p[j])
            # stop once the residual squared power is exhausted or below tol_p
            # (pj == 0 would divide by sqrt(0) below).
            if pj <= self.tol_p:
                break
            idx_sel[n] = j

            # new kernel column against the freshly selected center (from cache)
            a_new = self.kernel.col(j)  # (N,), cheap matvec

            # Newton basis function value at all grid points:
            #   v_n(x) = ( K(x, x_j) - sum_{l<n} v_l(x) v_l(x_j) ) / sqrt(pj)
            if n > 0:
                a_new = a_new - V[:, :n] @ V[j, :n]
            v_n = a_new / pj**0.5
            V[:, n] = v_n

            # incremental power update
            p = torch.clamp(p - v_n * v_n, min=0.0)
            n_used = n + 1
            pmax[n_used] = torch.sqrt(p.max())

            if self.verbose and (n % 50 == 0):
                print(f"  [greedy] n={n_used:4d}  sup Pow = {float(pmax[n_used]):.3e}")

        self.idx_ = idx_sel[:n_used]
        self.ctrs_ = Xd[self.idx_]
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
    """Evaluate the power function directly from a fixed center set.

    ``jitter`` is added to the Gram diagonal to stabilize the solve.
    """
    gram = kernel.eval(centers, centers)
    gram = gram + jitter * torch.eye(
        centers.shape[0], dtype=kernel.dtype, device=centers.device
    )
    cross = kernel.eval(query, centers)
    coefficients = torch.linalg.solve(gram, cross.T)
    residual = kernel.diagonal(query) - torch.sum(cross * coefficients.T, dim=1)
    return torch.sqrt(torch.clamp(residual, min=0.0))

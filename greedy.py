"""
P-greedy center selection for kernel interpolation.

The next center is placed where the power function
    Pow_P(x) = dist_H( K(.,x), span{ K(.,x_i) : x_i in P } )
is maximal over a fixed candidate grid -- the weak greedy of DeVore-Petrova-Wojtaszczyk
(arXiv:1204.2290) with gamma=1/2 on the dictionary of kernel translates, realized by the
exact argmax (which trivially meets the gamma=1/2 rule).  PGreedy maintains the residual
power at EVERY grid point via an incremental Newton basis (Pazouki-Schaback): O(N)/step,
O(N m^2) total, returning the full curve g_m = sup_x Pow_m(x) over m centers.  Device-
agnostic (allocates on the input's device) and dtype-threaded (default float64; float32
is faster on GPU but its power update p <- p - v_n^2 cancels badly as Pow -> 0).
"""

from __future__ import annotations
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        """g_m^lin = sup_x Pow_m(x) as a function of the number of centers m.

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


def design_figure(
    kernel,
    fit_grid: torch.Tensor,
    label: str,
    design_note: str,
    out: str,
    m_snap: int = 12,
    n_design: int = 128,
):
    """Two-panel P-greedy visualization shared by the 1-D kernel drivers:
      top    -- Pow_m(x) after m_snap centers + the next chosen point (argmax = the greedy rule),
      bottom -- the whole design (center position vs selection order), exposing the kernel's
                geometry (endpoint-clustered vs quasi-uniform, per `design_note`).
    `fit_grid` is passed in (its measure is kernel-specific; keeps greedy.py free of widths).
    """
    xq = torch.linspace(-1, 1, 4000, dtype=kernel.dtype).reshape(-1, 1)
    gr = PGreedy(kernel, max_iter=n_design, dtype=kernel.dtype).fit(fit_grid)
    ctrs = gr.ctrs_.reshape(-1).cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 8.0))
    P = gr.ctrs_[:m_snap]
    pw = power_function(kernel, P, xq).cpu().numpy()
    next_x = ctrs[m_snap]
    next_p = float(power_function(kernel, P, gr.ctrs_[m_snap : m_snap + 1])[0])
    ax = axes[0]
    ax.plot(
        xq.reshape(-1).cpu().numpy(),
        pw,
        color="C0",
        lw=1.6,
        label=rf"$\mathrm{{Pow}}_{{{m_snap}}}(x)$",
    )
    ax.plot(
        ctrs[:m_snap],
        np.zeros(m_snap),
        "|",
        color="0.35",
        ms=14,
        mew=1.5,
        label=f"{m_snap} chosen centers",
    )
    ax.plot(
        [next_x],
        [next_p],
        "*",
        color="C3",
        ms=16,
        label=r"next chosen point $=\arg\max\,\mathrm{Pow}$",
    )
    ax.axvline(next_x, color="C3", lw=0.8, ls=":")
    ax.set_xlabel(r"$x \in [-1,1]$")
    ax.set_ylabel(r"power function $\mathrm{Pow}_m(x)$")
    ax.set_title(rf"{label}: the greedy rule (argmax of $\mathrm{{Pow}}_{{{m_snap}}}$)")
    ax.set_ylim(0, 1.32 * max(float(pw.max()), next_p))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    order = np.arange(1, len(ctrs) + 1)
    ax.scatter(ctrs, order, c=order, cmap="viridis", s=14)
    ax.set_xlabel(r"center position $x$")
    ax.set_ylabel(r"selection order $n$")
    ax.set_title(rf"{label}: greedy design ({design_note})")
    ax.set_xlim(-1.03, 1.03)
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        "P-greedy point selection: each chosen point is the argmax of the power function"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.02)
    print(f"  figure saved -> {out}")

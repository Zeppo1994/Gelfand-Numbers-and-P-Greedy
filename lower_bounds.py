"""Shared helpers for P-greedy sampling curves and covariance-tail lower bounds."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from greedy import PGreedy

LOWER_BOUND_SYMBOL = "c̲ₙ"


def box_grid(
    n: int,
    d: int,
    dtype: torch.dtype,
    device: str = "cpu",
    kind: str = "chebyshev",
    domain=(-1.0, 1.0),
) -> torch.Tensor:
    """Return a deterministic candidate grid on the requested box."""
    n = int(n)
    d = int(d)
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive")

    lo, hi = domain
    if not lo < hi:
        raise ValueError("domain must satisfy lo < hi")

    if kind not in {"chebyshev", "equidistant", "periodic"}:
        raise ValueError(f"unknown grid kind: {kind}")

    if d == 1:
        if kind == "chebyshev":
            angles = torch.linspace(0, np.pi, n, dtype=dtype, device=device)
            unit_grid = 0.5 * (1.0 - torch.cos(angles))
            return (lo + (hi - lo) * unit_grid)[:, None]
        count = n + 1 if kind == "periodic" else n
        grid = torch.linspace(lo, hi, count, dtype=dtype, device=device)
        return grid[:n, None]

    engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=0)
    unit_grid = engine.draw(n).to(device=device, dtype=dtype)
    return lo + (hi - lo) * unit_grid


def fit_p_greedy(kernel, *, d: int = 1, max_iter: int, sel_grid: int) -> PGreedy:
    """Fit P-greedy using the kernel's domain and candidate-grid convention."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = box_grid(
        sel_grid,
        d,
        kernel.dtype,
        device,
        kind=kernel.grid_kind,
        domain=kernel.domain,
    )
    return PGreedy(kernel, max_iter=max_iter, dtype=kernel.dtype).fit(grid)


def log_spaced_ints(hi: int, count: int) -> np.ndarray:
    """Return distinct approximately log-spaced integers in [1, hi]."""
    hi = int(hi)
    count = int(count)
    if hi < 1 or count < 1:
        raise ValueError("hi and count must be positive")
    return np.unique(np.round(np.geomspace(1, hi, count)).astype(int))


def clipped_prefix(indices, values, floor):
    """Return a curve ending exactly at its first crossing of floor."""
    indices = np.asarray(indices)
    values = np.asarray(values)
    below = np.flatnonzero(values <= floor)
    if below.size == 0:
        return indices, values
    stop = int(below[0])
    clipped = values[: stop + 1].copy()
    clipped[-1] = floor
    return indices[: stop + 1], clipped


def finalize_figure(fig, out) -> Path:
    """Lay out, save, and close one figure."""
    from matplotlib import pyplot as plt

    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        output,
        dpi=130,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    return output


def add_prefixed_result(target: dict, prefix: str, result: dict) -> None:
    """Add result entries to target with a shared prefix."""
    target.update({f"{prefix}_{key}": value for key, value in result.items()})


def plot_sampling_estimate(
    ax,
    n,
    values,
    *,
    color="C0",
    label=r"$g_m^{\mathrm{lin}}$ (P-greedy estimate)",
    **kwargs,
):
    """Plot a P-greedy estimate with the shared manuscript style."""
    return ax.loglog(
        n, values, "o-", ms=3.5, lw=1.6, color=color, label=label, **kwargs
    )[0]


def plot_gelfand_lower_bound(
    ax,
    n,
    lower,
    *,
    color="C1",
    label=f"lower bound {LOWER_BOUND_SYMBOL} ≤ cₙ",
):
    """Plot a lower-bound curve with the shared manuscript style."""
    return ax.loglog(n, lower, "-", color=color, lw=1.5, label=label)[0]


def sampling_vs_lower_bound(
    kernel,
    *,
    d: int = 1,
    m_max: int = 1000,
    sel_grid: int = 20_000,
    n_cap: int | None = None,
    lower_bound_kwargs: dict | None = None,
):
    """Compute one P-greedy sampling curve and a covariance-tail lower bound.

    The sampling curve is a finite-grid numerical surrogate. The lower curve
    comes from the kernel's gelfand_lower_tails method and is independent
    of the greedy design.
    """
    greedy = fit_p_greedy(kernel, d=d, max_iter=m_max, sel_grid=sel_grid)
    n_used = greedy.n_
    if n_used < 2:
        raise RuntimeError("P-greedy stopped before producing a comparison curve")

    # Exclude the final update, which can be exactly zero on an exhausted grid.
    g_curve = greedy.g_curve().cpu().numpy()
    sampling_n = log_spaced_ints(n_used - 1, 60)
    sampling = g_curve[sampling_n]

    cap = min(n_used - 1, int(n_cap) if n_cap is not None else n_used - 1)
    if cap < 1:
        raise ValueError("n_cap must allow at least one positive width index")
    kwargs = {} if lower_bound_kwargs is None else dict(lower_bound_kwargs)
    tails = np.asarray(kernel.gelfand_lower_tails(cap, **kwargs), dtype=np.float64)
    if tails.shape != (cap + 1,):
        raise ValueError("gelfand_lower_tails must return values for n=0,...,n_cap")
    lower_n = np.arange(1, cap + 1)
    lower = np.sqrt(np.clip(tails[lower_n], 0.0, None))
    sampling_at_lower_n = np.exp(
        np.interp(np.log(lower_n), np.log(sampling_n), np.log(sampling))
    )
    ratio = np.divide(
        sampling_at_lower_n,
        lower,
        out=np.full_like(lower, np.nan),
        where=lower > 0.0,
    )
    return {
        "sampling_n": sampling_n,
        "sampling": sampling,
        "lower_n": lower_n,
        "lower": lower,
        "sampling_at_lower_n": sampling_at_lower_n,
        "ratio": ratio,
        "n_used": n_used,
        "d": d,
    }

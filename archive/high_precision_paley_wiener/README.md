# Archived: high-precision (mpmath) Gelfand widths for the band-limited kernel

This folder holds the **arbitrary-precision** extension of the Paley–Wiener experiment: an
mpmath re-implementation of the Gelfand-width minimax that recovers the true widths `c_n` past
the float64 cliff, where a band-limited Gram (numerical rank ≈ `N_eff = 2c/π`) floors at
`~sqrt(eps)·sqrt(K(x,x)) ≈ 1e-6`.

It was **retired from the live pipeline** on 2026-07-14. The live `paley_wiener.py` now runs the
float64 comparison only. Everything needed to reproduce the hp figure is preserved here.

## Why it was archived (the honest reason)

The hp run is expensive (pure-Python mpmath eigensolves) and, as configured, it **only earned its
cost in one of the three bandwidths.** Measured against each bandwidth's float64 floor `τ`:

| panel      | float64 floor τ | hp bracket reaches | points below floor | verdict |
|------------|-----------------|--------------------|--------------------|---------|
| N_eff = 20 | 9.4e-7          | 1.7e-5             | 0 / 10             | hp sits *above* the floor — float64 already resolves it |
| N_eff = 40 | 1.3e-6          | **3.6e-16**        | **5 / 11**         | the real showcase: ~10 orders below the floor, tight bracket (1.36) |
| N_eff = 80 | 1.9e-6          | 3.5e-4             | 0 / 9              | hp *above* the floor, loosest bracket (2.44) |

The cause is the choice of `n`-abscissae, not the method: `HP_NLISTS[40]` runs to `n=70` (30 past
`N_eff`, deep down the super-exponential cliff), while the `N_eff=20` and `80` lists stop ~10 past
their cliffs and never cross below the float64 floor. Two of the three hp panels therefore just
overlaid the float64 bracket and bought nothing.

Rather than keep paying for two redundant hp computations (or extend the `n`-lists and re-tune `M`
/ `dps` deep into the cliff), the experiment was simplified to the float64-only figure. Keep this
in mind before re-enabling: **hp is only worth it if the `n`-list is pushed far enough past
`N_eff` that the true `c_n` drops below `τ`** — see the `N_eff=40` panel for what "worth it" looks
like.

## Contents

| file                  | what it is |
|-----------------------|------------|
| `widths_hp.py`        | `gelfand_widths_hp` + `_golden_max` — the mpmath minimax (extracted from `widths.py`). |
| `kernels_hp.py`       | `PaleyWienerSincKernelHP` (subclass adding `kfun_mp` + `prolate_spectrum_hp`) and `_gauss_legendre_hp` (extracted from `kernels.py`). |
| `paley_wiener_hp.py`  | the driver: float64 `g_m^lin` / `c_n` bracket **plus** the hp bracket continuing past the cliff, over `N_eff ∈ {20,40,80}`. |
| `paley_wiener_hp.npz` | cached hp brackets (the slow part). Delete or pass `recompute_hp=True` to rebuild. |

`kernels_hp.py` imports the live `PaleyWienerSincKernel` from the parent package and adds only the
hp members; `widths_hp.py` is self-contained (numpy + mpmath). The driver pulls the float64 half
(`widths`, `greedy`) from the parent package and the hp half from these local modules.

## Build / run

Needs the `PyTorch` conda env (numpy, torch, scipy, matplotlib, **mpmath**) and the parent package
on `PYTHONPATH` alongside this folder. From the repo root:

```bash
conda run -n PyTorch env \
  PYTHONPATH=/LOCAL/sebne/sampling_number_greedy:/LOCAL/sebne/sampling_number_greedy/archive/high_precision_paley_wiener \
  python /LOCAL/sebne/sampling_number_greedy/archive/high_precision_paley_wiener/paley_wiener_hp.py
```

This writes `paley_wiener.png` (the 3-bandwidth width figure with the hp overlay) and
`paley_wiener_points.png` into the current directory. The hp brackets load from
`paley_wiener_hp.npz` if the `HP_NLISTS` match; otherwise they recompute (slow — minutes — and the
mpmath eigensolves are single-threaded).

> Shared-machine note: the mpmath solve is CPU-heavy and single-threaded. Run it **one at a time**,
> in the foreground, and don't stack it with other conda jobs.

To rebuild the cache from scratch, call `rates_figure(recompute_hp=True)` (edit the `__main__`
block) or delete `paley_wiener_hp.npz`.

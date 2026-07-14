# Sampling numbers vs Gelfand widths — a kernel-RKHS numerical study

Numerical assessment of the comparison theorems in **`MAIN_Sampl_vs_Gelfand.tex`**
(Neumayer–Pozharska–Ullrich, *sampling numbers versus Gelfand widths for optimal
recovery in the uniform norm*).

## The question

For the unit ball of a bounded-kernel RKHS `H(K)` on `D=[-1,1]^d`, recovered in the
**uniform norm**, the paper compares two quantities as functions of the budget:

* **`g_m^lin`** — the *linear sampling numbers* `= inf_{|P|=m} ‖Pow_P‖_∞`, the best
  worst-case error from `m` point samples. By Prop. `gm` these are realized by
  **P-greedy points** — i.e. the **weak greedy with `γ=1/2`** ("place the next center
  where the power function is ≥ half its sup") on the dictionary of translates `{K(·,x)}`.
* **`c_n`** — the *Gelfand widths* `= d_n(𝒦)_H`, the Kolmogorov `n`-width of the
  translate set, i.e. the best error from `n` *arbitrary* linear measurements.

Since point evaluations are a restricted class of functionals, `c_n ≤ g_m^lin` always.
The general bound (KPUU) only gives `g_{2n} ≤ C·√n·c_n`. **The paper's theorem removes
the `√n`** for a bounded-kernel RKHS: the whole regularly-varying decay scale transfers,
so `g_m^lin ≲ c_n` and the ratio `g_m/c_n` stays **bounded**.

## What the code shows

* `g_m^lin` and `c_n` seem to run **parallel** 
* The **ratio `g_m/c_n` remains bounded** (no `√n` growth) — Matérn `ν=3/2` median 2.31 (1D),
  1.25 (`d=3`); periodic `H^m_mix` 1.47–2.14 across `m=1..3`, `d=2,3`; band-limited sinc 0.98 (flat
  regime). This is the theorem; the KPUU `√n` is never needed. Ratios are quoted against `c_n⁺`;
  the certified/exchanged `c_n⁺` is tighter than the old estimate, so they read slightly *higher*
  than before — closer to the true `g_m/c_n`.
* In `legendre.py`, `√n·c_n` sits a full half-power above `g_m^lin` (the closed gap); in
  `matern.py` the same point is made dimension-robustly by the flat ratio.

## Files

| file                 | contents                                                              |
|----------------------|-----------------------------------------------------------------------|
| `kernels.py`         | Legendre Mercer `K_s`, Matérn (half-integer `ν`, any `d`), periodic mixed-Sobolev `H^m_mix`, Paley–Wiener sinc `K_c` (band-limited / prolate); stationary kernels expose the rigorous RKHS modulus `dist_bound` the `c_n⁺` certificate needs |
| `greedy.py`          | PyTorch P-greedy (incremental Newton-basis power function)            |
| `widths.py`          | `g_m^lin` (P-greedy sup power), `c_n` (numerical Kolmogorov width: IRLS minimax + exchange + certified/estimated sup), driver |
| `legendre.py`        | Legendre example → `legendre.png`; the P-greedy design → `legendre_points.png` |
| `matern.py`          | Matérn `ν=3/2` in `d=1,3`, bounded-ratio check → `matern.png`; the P-greedy design → `matern_points.png` |
| `periodic_mixed.py`  | periodic mixed-Sobolev `H^m_mix([0,1]^d)`, `d=2,3`, `m=1,2,3` → `periodic_mixed.png`, `periodic_mixed_points.png` |
| `paley_wiener.py`    | band-limited Paley–Wiener/prolate kernel: flat-then-cliff `g_m^lin` and `c_n` at `N_eff=2c/π` (float64; the retired mpmath past-the-cliff run lives in `archive/high_precision_paley_wiener/`) → `paley_wiener.png`; the P-greedy design → `paley_wiener_points.png` |

## Usage

```bash
python legendre.py       # Legendre s=2: sampling-vs-Gelfand figure + P-greedy design
python matern.py         # Matérn ν=3/2 (d=1,3): bounded-ratio check + P-greedy design
python periodic_mixed.py # periodic mixed-Sobolev H^m_mix, d=2,3 -- ratio + rate overlay
python paley_wiener.py   # band-limited kernel: the N_eff cliff in float64 + design
```

Each driver writes `<kernel>.png` (the `g_m^lin` vs `c_n` comparison) and, where applicable,
`<kernel>_points.png` (the P-greedy design).

## Method notes

* **Candidate grid is kernel-specific** (`kernel.grid_kind`, consumed by `widths.box_grid`). A
  Chebyshev grid encodes the *arcsine/equilibrium* measure — right only for a **boundary-concentrated
  Mercer** kernel (Legendre, `grid_kind="chebyshev"`). A **stationary** kernel (Matérn, sinc,
  periodic) has no endpoint preference, so its measure is **uniform** (scrambled Sobol,
  `grid_kind="uniform"`, any `d`).
* **`g_m^lin`** = the running `sup_x Pow_m(x)` of the P-greedy design (the strong argmax trivially
  meets the `γ=1/2` weak rule). Its plateau-then-drop staircase for Matérn is the signature of a
  *stationary* kernel (dyadic gap-bisection, quasi-uniform centers), versus the smooth power law of
  the endpoint-clustered Legendre design. Each driver's `points_figure()` (via `greedy.design_figure`)
  plots the chosen point as the argmax of the power function and the resulting design (position vs.
  selection order) — endpoint-clustered for Legendre, quasi-uniform for Matérn.
* **`c_n`** = the Kolmogorov width of the translate set by a **reweighted-SVD (IRLS) minimax with
  a Remez exchange step**: kernel-PCA on the reweighted Gram, shift weight to the worst point. The
  linear reweight `w ← w·(r² + floor·r²_max)` is the only rule, with `floor=1` as a **damping
  constant** — the smooth step keeps the best-iterate subspace stable off-grid.
  Returned as a **bracket `[c_n⁻, c_n⁺]`**:
    * `c_n⁻` (lower) — **rigorous** (weighted-average residual `= Σ_{k>n} λ_k(C_p)`, weak duality),
      so `c_n⁻ ≤ c_n` always (up to the `r_n = 3n+100` truncation, `≤ 1e-4` relative).
    * `c_n⁺` (upper) — on a **1D grid with a `dist_bound` modulus** (Matérn, sinc, periodic) a
      **branch-and-bound certificate**: each cell bounded by `r(center) + dist_bound(halfwidth)`,
      bisect what exceeds the incumbent, prune the rest. Where no modulus exists — **Legendre** (endpoint
      modulus diverges) and **`d>1`** (cells explode) — `c_n⁺` is a **numerical estimate**: L-BFGS-B
      multistart from two polished seed sets + a disagreement **self-check**, plus the **endpoint
      sweep** (1D Mercer) for the boundary spike.
    * **exchange** (stationary default): the off-grid residual peaks rejoin the dual set, the basis
      is extended *exactly* by their translates (Nyström update, no re-eigendecomposition), a short
      warm IRLS re-optimizes; `max` of lowers / `min` of uppers is kept. For **Legendre** exchange is auto-off (the spike re-emerges at a new offset);
      instead the `c_n` grid gets a **geometric endpoint ladder** (`box_grid(edge_ladder=120)`).
    * monotone envelopes: `c_n⁺ ← min_{k≤n} c_k⁺` (**subspace nesting**; certified entries stay
      certificates) and `c_n⁻ ← max_{m≥n} c_m⁻`.
  Kernel-agnostic (needs `kernel.eval`; `dist_bound` → certificate, `feature_map` → Mercer speed,
  `eval_grad` → multistart gradients). Stable up to the Gram's numerical rank; algebraic
  Legendre/Matérn reach hundreds.
* **P-greedy engine** matches the standard VKOGA implementation to `~1e-8`; it uses the
  efficient incremental power update (`O(N·m²)` total, cached kernel columns), is
  device-agnostic (a CUDA grid runs on GPU), and dtype-threaded.
* **Periodic mixed-Sobolev test** (`periodic_mixed.py`) probes the *mixed-smoothness* regime: the
  RKHS `H^m_mix([0,1]^d)`, reproducing kernel the `d`-fold tensor of
  `k_1(s,t)=1+(-1)^{m-1}/(2m)!·B_{2m}(|s-t|)` (Bernoulli polynomial; Berlinet–Thomas-Agnan p.318),
  `d=2,3`, `m=1,2,3`. It confirms the **bounded ratio** there too, with the literature rate
  `c_n ∼ n^{-(m-1/2)}(log n)^{(d-1)m}` overlaid as an *asymptotic guide* (constant fitted to `c_n`
  on the post-hump tail). The log factor is far from developed at reachable `n` — which is why the
  direct `g_n/c_n` ratio, not the rate line, is the trustworthy comparison in higher `d`.

## Caveats

* **Keep `float64`** (with one audited exception). The incremental power update
  `p ← p − v_n²` cancels badly as `Pow → 0`; `float32` (fast on GPU) is unsafe once the
  tracked widths drop below ~1e-4, and would truncate these curves early. `s=2` keeps
  `Pow²` above the `float64` floor out to `m=1000`; `s ≥ 4` stops near `m≈200`. The **only**
  place `float32` is used is the `H^1_mix` (`m=1`) sampling curve in `periodic_mixed.py`,
  whose error decays slowly (`n^{-1/2}`, staying above ~1e-2 out to `n≈4000`) and never
  approaches the float32 floor — validated to agree with `float64` to **<0.6%** over the
  whole curve, in exchange for a ~16× GPU speedup that buys the large node budget. All
  Gelfand-width computations always run in `float64`.
* **Know which `c_n⁺` you are looking at.** On 1D stationary kernels it is a **branch-and-bound
  certificate** (rigorous up to `certify_tol` and float64 rounding); it reverts to a **resolved
  estimate, not a proof**, exactly where certification is impossible:
  * **Legendre** — no finite endpoint modulus; the estimate leans on the endpoint sweep + the
    `edge_ladder` grid (which fixed the historical `~20%` undershoot: the peak sits within `~3·10⁻⁷`
    of `±1`, the innermost plain-Chebyshev node at `~1/N`).
  * **`d>1`** — multistart L-BFGS-B: too few starts silently under-resolves the sup, so `c_n⁺` can
    drop *below* the true value. Mitigations (dual-sampler max, `~3n` starts, disagreement warning,
    exchange re-optimizing against the found peaks) make this unlikely, not impossible — the failure
    is real (on 1D Matérn the certificate caught a lean multistart 10% low). The `d=3` Matérn is the
    stress case, needing a denser (`8000`-point) grid.
  * **at the float64 noise floor** (band-limited kernel past `N_eff`) no budget certifies;
    `gelfand_widths` warns and falls back to its dense-scan best (the floored values carry no
    information anyway).
  `refine=False` reverts to the grid-max upper bound (no certificate/exchange; under-resolves at
  large `n`).

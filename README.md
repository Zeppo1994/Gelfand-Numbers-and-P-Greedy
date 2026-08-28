# Sampling numbers vs Gelfand widths

Numerical experiments for optimal recovery of bounded-kernel RKHS unit balls in the
uniform norm. The code compares linear sampling numbers with Gelfand widths for
Legendre, Matérn, periodic mixed-Sobolev, and Paley–Wiener kernels.

## Quantities

- `g_m^lin`: best worst-case error using `m` point evaluations.
- `c_n`: best worst-case error using `n` arbitrary linear measurements.

The code estimates `g_m^lin` with the power function of one P-greedy design on a
finite candidate grid. These values are constructive upper surrogates, not solutions
of the global point-set optimization. For equal budgets, `c_n ≤ g_n^lin`.

The reported Gelfand-width interval consists of a rigorous lower value `c_n^-` and an
upper value `c_n^+`. The upper value is certified only where the returned
`cn_certified` mask is true; elsewhere it is an off-grid numerical estimate.

## Setup

Requires Python 3.10 or newer. The reproducible environment uses Python 3.12.

```bash
conda env create -f environment.yml
conda activate sampling-numbers
pytest -q
```

CUDA is used automatically for the large P-greedy grids. Width computations run in
float64 on the CPU. All experiments support CPU-only execution, but the publication
settings are expensive; `periodic_mixed.py` in particular needs substantial memory.

## Experiments

```bash
python legendre.py
python matern.py
python periodic_mixed.py
python paley_wiener.py
```

| driver | output |
|---|---|
| `legendre.py` | `figures/legendre.png`, `figures/legendre_points.png` |
| `matern.py` | `figures/matern.png`, `figures/matern_points.png` |
| `periodic_mixed.py` | `figures/periodic_mixed.png`, `figures/periodic_mixed_points.png` |
| `paley_wiener.py` | `figures/paley_wiener.png`, `figures/paley_wiener_points.png` |

The Matérn experiment is supplemental. Legendre and Matérn use `[-1,1]^d`, the
periodic experiment uses `[0,1]^d`, and Paley–Wiener uses `[-1,1]`.
Both Legendre figures use Mercer truncation `M=24,000` and a 20,000-point
Chebyshev candidate grid.

## Publication run

Run the manuscript figures sequentially in the foreground:

```bash
python run_manuscript_figures.py --output-dir runs/manuscript
```

The runner writes PNGs to `<output-dir>/figures/` and keeps numerical arrays,
`run.log`, and `status.json` in the output directory. Resume completed or interrupted stages with:

```bash
python run_manuscript_figures.py --output-dir runs/manuscript --resume
```

External schedulers or process managers can wrap this command when background execution
is needed.

## Implementation

- `kernels.py`: kernel definitions, grids, feature maps, gradients, and available
  modulus or spectral-tail helpers.
- `greedy.py`: device-agnostic incremental P-greedy algorithm.
- `widths.py`: grid construction, Gelfand-width minimax, certification, exchange, and
  the shared sampling-vs-width driver.
- `run_manuscript_figures.py`: resumable publication workflow.

P-greedy uses an incremental Newton-basis power update. Candidate grids are
kernel-specific: Chebyshev for the boundary-concentrated Legendre kernel and uniform
Sobol grids for stationary kernels.

The Gelfand-width computation uses reweighted SVD with optional Remez exchange.
`compress_irls=True`, the driver default, retains `min(N, 3n+100)` dominant
coordinates. Set `compress_irls=False` to use the full coordinates employed by the
manuscript algorithm:

```python
import legendre
legendre.comparison_figure(compress_irls=False)
```

Certified upper values use a branch-and-bound residual supremum when the kernel
provides a rigorous cell modulus. Legendre uses a numerical supremum estimate, as do
periodic kernels in dimension greater than one. The periodic spectral tail is exact at
closed Fourier shells and provides an independent reference.

## Numerical caveats

- Keep Gelfand-width computations in float64. The incremental power update also uses
  float64 except for the audited `H^1_mix` sampling curve in `periodic_mixed.py`.
- Values near the float64 cancellation floor are numerical floors, not resolved
  widths. This is most visible past the Paley–Wiener effective dimension.
- `refine=False` returns a grid-maximum estimate without certification or exchange.
- Check `cn_certified` before treating a plotted upper edge as a proof.

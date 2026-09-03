# P-greedy sampling estimates and Gelfand-width lower bounds

Numerical RKHS experiments comparing grid-based P-greedy sampling estimates with
computable lower bounds for Gelfand widths.

For the same measurement budget $n$, the exact continuum quantities satisfy

\[
\underline{c}_n \leq c_n \leq g_n^{\mathrm{lin}} \leq P(X_n),
\]

where $P(X_n)$ is the worst-case power-function error of the selected point set.
The P-greedy curve in the plots is the maximum of the power function on a
finite candidate grid. That grid maximum is a numerical surrogate for $P(X_n)$,
not a certified continuum upper bound or a solution of the global point-set
problem.

The four experiments are:

- Legendre Mercer kernels on $[-1,1]$;
- Matérn kernels on $[-1,1]^d$;
- periodic mixed-Sobolev kernels on $[0,1]^d$;
- Paley-Wiener sinc kernels on $[-1,1]$.

## Run

Python 3.10 or newer is required. Install the runtime dependencies with

```bash
python -m pip install -r requirements.txt
```

Run experiments individually with

```bash
python legendre.py
python matern.py
python periodic_mixed.py
python paley_wiener.py
```

Each driver writes its comparison and point-design figures under `figures/`:

| Driver | Comparison | Point design |
| --- | --- | --- |
| `legendre.py` | `legendre.png` | `legendre_points.png` |
| `matern.py` | `matern.png` | `matern_points.png` |
| `periodic_mixed.py` | `periodic_mixed.png` | `periodic_mixed_points.png` |
| `paley_wiener.py` | `paley_wiener.png` | `paley_wiener_points.png` |

The ignored `figures/` directory contains generated output, not source files.

## What is computed

Every lower curve is a covariance-eigenvalue tail for a probability measure on
the domain:

- Legendre uses normalized Lebesgue measure and the exact infinite Mercer tail.
  P-greedy evaluates the infinite kernel for integer `s` by a partial-fraction
  Green-kernel formula built from a few complex Legendre functions on the
  candidate grid; there is no feature truncation. The large-`m` flattening of
  the `s=3` P-greedy curve is a float64 cancellation floor, not an asymptotic
  feature.
- Matérn uses Gauss-Legendre quadrature in one dimension and an equal-weight
  Sobol discrete measure in three dimensions.
- Periodic mixed Sobolev uses the complex Fourier covariance tail, which is a lower
  bound for the real-space width and agrees at complete sine/cosine shells.
- Paley-Wiener uses the covariance-eigenvalue tail of a Gauss-Legendre
  probability measure on $[-1,1]$. Each finite discrete tail is an
  exact-arithmetic lower bound and converges to the normalized Slepian tail as
  the quadrature is refined.

The lower-bound constructions are rigorous in exact arithmetic; the plotted
float64 eigensolver results are not interval certificates. A conservative
eigenvalue shift and per-experiment resolution floors keep unresolved values
out of the interpretation.

The implementation is concentrated in `greedy.py`, `kernels.py`,
`lower_bounds.py`, and the small experiment drivers.

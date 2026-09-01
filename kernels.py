"""Kernels used by the P-greedy and covariance-tail experiments."""

from __future__ import annotations

import heapq
import math

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.special import bernoulli, comb, digamma, factorial


def _discrete_covariance_lower_tails(
    covariance: np.ndarray,
    n_max: int,
    *,
    exact_trace: float,
) -> np.ndarray:
    """Return covariance tails after one shared float64 safety adjustment.

    In exact arithmetic, every positive discrete probability measure produces
    a rigorous lower bound. The subtraction below avoids promoting tiny
    positive eigensolver artifacts to resolved lower values; the exact trace
    is restored at index zero.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not 0 <= n_max < covariance.shape[0]:
        raise ValueError("n_max must be smaller than the covariance size")

    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    error = (
        32.0
        * covariance.shape[0]
        * np.finfo(np.float64).eps
        * max(float(eigenvalues[-1]), float(exact_trace))
    )
    eigenvalues = np.clip(eigenvalues - error, 0.0, None)
    tails = np.cumsum(eigenvalues, dtype=np.float64)[::-1]
    tails[0] = exact_trace
    return tails[: n_max + 1]


def _complex_legendre_p(eigenvalue: complex, points: np.ndarray) -> np.ndarray:
    """Evaluate P_nu where nu(nu+1)=eigenvalue on real points in [-1, 1]."""
    points = np.asarray(points, dtype=np.float64)
    if np.any((points < -1.0) | (points > 1.0)):
        raise ValueError("Legendre arguments must lie in [-1, 1]")

    values = np.empty(points.shape, dtype=np.complex128)
    values[points == 1.0] = 1.0
    values[points == -1.0] = np.nan
    interior = (points > -1.0) & (points < 1.0)
    if not np.any(interior):
        return values

    unique, inverse = np.unique(points[interior], return_inverse=True)
    descending = unique[::-1]
    distance_from_one = 1.0 - descending[0]
    epsilon = min(1e-10, 0.5 * distance_from_one)
    start = 1.0 - epsilon
    initial = np.array(
        [1.0 - 0.5 * eigenvalue * epsilon, 0.5 * eigenvalue],
        dtype=np.complex128,
    )
    solution = solve_ivp(
        lambda x, state: np.array(
            [
                state[1],
                (2.0 * x * state[1] - eigenvalue * state[0])
                / (1.0 - x * x),
            ]
        ),
        (start, descending[-1]),
        initial,
        t_eval=descending,
        method="DOP853",
        rtol=2e-12,
        atol=2e-13,
    )
    if not solution.success:
        raise RuntimeError(f"complex Legendre solve failed: {solution.message}")

    values[interior] = solution.y[0, ::-1][inverse]
    return values


class LegendreMercerKernel:
    """Infinite Legendre Mercer kernel with algebraic eigenvalue decay.

    Kernel evaluation uses a partial-fraction Green-kernel formula for the
    Legendre operator. No feature truncation or grid-by-mode cache is used.
    """

    domain = (-1.0, 1.0)
    grid_kind = "chebyshev"

    def __init__(
        self,
        s: float = 2.0,
        dtype: torch.dtype = torch.float64,
    ):
        self.s = float(s)
        order = round(self.s)
        if order < 2 or abs(self.s - order) > 1e-12:
            raise ValueError("s must be an integer greater than or equal to 2")
        self.order = int(order)
        self.dtype = dtype

        roots = np.exp(
            1j
            * np.pi
            * (2.0 * np.arange(self.order, dtype=np.float64) + 1.0)
            / self.order
        )
        partial_fractions = 1.0 / (
            self.order * roots ** (self.order - 1)
        )
        degrees = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * roots))
        self._roots = roots
        self._resolvent_coefficients_np = (
            -np.pi
            * partial_fractions
            / (2.0 * np.sin(np.pi * degrees))
        )
        self._endpoint_diagonal = float(
            np.real(
                -0.5
                * np.sum(
                    partial_fractions
                    * (digamma(-degrees) + digamma(degrees + 1.0))
                )
            )
        )

    def _resolvent_basis(self, points: np.ndarray):
        plus = np.stack(
            [_complex_legendre_p(root, points) for root in self._roots]
        )
        if np.allclose(points, -points[::-1], rtol=0.0, atol=2e-14):
            minus = plus[:, ::-1].copy()
        else:
            minus = np.stack(
                [_complex_legendre_p(root, -points) for root in self._roots]
            )
        return plus, minus

    def eval(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        x = X.reshape(-1).detach().cpu().numpy()
        y = Y.reshape(-1).detach().cpu().numpy()
        x_plus, x_minus = self._resolvent_basis(x)
        y_plus, y_minus = self._resolvent_basis(y)
        lower = x[:, None] <= y[None, :]
        matrix = np.zeros((x.size, y.size), dtype=np.complex128)
        for coefficient, px, mx, py, my in zip(
            self._resolvent_coefficients_np,
            x_plus,
            x_minus,
            y_plus,
            y_minus,
        ):
            matrix += coefficient * np.where(
                lower,
                mx[:, None] * py[None, :],
                px[:, None] * my[None, :],
            )
        endpoint_pairs = (
            (x[:, None] == y[None, :])
            & (np.abs(x[:, None]) == 1.0)
        )
        matrix[endpoint_pairs] = self._endpoint_diagonal
        return torch.as_tensor(
            np.real(matrix),
            dtype=self.dtype,
            device=X.device,
        )

    def diagonal(self, X: torch.Tensor) -> torch.Tensor:
        points = X.reshape(-1).detach().cpu().numpy()
        plus, minus = self._resolvent_basis(points)
        diagonal = np.real(
            np.sum(
                self._resolvent_coefficients_np[:, None] * plus * minus,
                axis=0,
            )
        )
        diagonal[np.abs(points) == 1.0] = self._endpoint_diagonal
        return torch.as_tensor(
            diagonal,
            dtype=self.dtype,
            device=X.device,
        )

    def prepare(self, Xd: torch.Tensor) -> None:
        self._Xd = Xd.reshape(-1, 1)
        points = self._Xd[:, 0].detach().cpu().numpy()
        if np.any(np.diff(points) < 0.0):
            raise ValueError(
                "the low-memory Legendre path requires a sorted grid"
            )
        plus, minus = self._resolvent_basis(points)
        device = self._Xd.device
        self._resolvent_coefficients = torch.as_tensor(
            self._resolvent_coefficients_np,
            dtype=torch.complex128,
            device=device,
        )
        self._p_plus = torch.as_tensor(
            plus,
            dtype=torch.complex128,
            device=device,
        )
        self._p_minus = torch.as_tensor(
            minus,
            dtype=torch.complex128,
            device=device,
        )
        diagonal = torch.sum(
            self._resolvent_coefficients[:, None]
            * self._p_plus
            * self._p_minus,
            dim=0,
        ).real
        endpoint_mask = torch.abs(self._Xd[:, 0]) == 1.0
        diagonal[endpoint_mask] = self._endpoint_diagonal
        self._diag = diagonal.to(self.dtype)

    def diag_grid(self) -> torch.Tensor:
        return self._diag

    def col(self, j: int) -> torch.Tensor:
        j = int(j)
        column = torch.empty(
            self._Xd.shape[0],
            dtype=self.dtype,
            device=self._Xd.device,
        )
        column[: j + 1] = torch.sum(
            self._resolvent_coefficients[:, None]
            * self._p_minus[:, : j + 1]
            * self._p_plus[:, j, None],
            dim=0,
        ).real.to(self.dtype)
        if j + 1 < self._Xd.shape[0]:
            column[j + 1 :] = torch.sum(
                self._resolvent_coefficients[:, None]
                * self._p_minus[:, j, None]
                * self._p_plus[:, j + 1 :],
                dim=0,
            ).real.to(self.dtype)
        column[j] = self._diag[j]
        return column

    def gelfand_lower_tails(self, n_max: int) -> np.ndarray:
        """Return safely resolved squared covariance-tail lower bounds."""
        n_max = int(n_max)
        if n_max < 0:
            raise ValueError("n_max must be nonnegative")
        cutoff = max(200_000, 200 * (n_max + 1))
        degrees = np.arange(cutoff, dtype=np.float64)
        eigenvalues = 1.0 / (
            1.0 + (degrees * (degrees + 1.0)) ** self.order
        )
        partial_tails = 0.5 * np.cumsum(
            eigenvalues[::-1], dtype=np.float64
        )[::-1]
        allowance = (
            32.0 * cutoff * np.finfo(np.float64).eps * partial_tails
        )
        return np.maximum(
            partial_tails[: n_max + 1] - allowance[: n_max + 1],
            0.0,
        )

class MaternKernel:
    """Half-integer Matérn kernel on [-1,1]^d."""

    domain = (-1.0, 1.0)
    grid_kind = "equidistant"

    def __init__(
        self,
        nu: float = 1.5,
        ell: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ):
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("nu must be one of 0.5, 1.5, 2.5")
        self.nu = float(nu)
        self.ell = float(ell)
        self.a = math.sqrt(2.0 * self.nu) / self.ell
        self.dtype = dtype

    def _corr(self, distance: torch.Tensor) -> torch.Tensor:
        scaled = self.a * distance
        exponential = torch.exp(-scaled)
        if self.nu == 0.5:
            return exponential
        if self.nu == 1.5:
            return (1.0 + scaled) * exponential
        return (1.0 + scaled + scaled * scaled / 3.0) * exponential

    def eval(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = X.reshape(X.shape[0], -1).to(self.dtype)
        Y = Y.reshape(Y.shape[0], -1).to(self.dtype)
        return self._corr(torch.cdist(X, Y))

    def diagonal(self, X: torch.Tensor) -> torch.Tensor:
        return torch.ones(X.shape[0], dtype=self.dtype, device=X.device)

    def prepare(self, Xd: torch.Tensor) -> None:
        self._Xd = Xd.reshape(Xd.shape[0], -1).to(self.dtype)
        self._diag = torch.ones(
            self._Xd.shape[0], dtype=self.dtype, device=self._Xd.device
        )

    def diag_grid(self) -> torch.Tensor:
        return self._diag

    def col(self, j: int) -> torch.Tensor:
        distances = torch.cdist(self._Xd, self._Xd[j : j + 1])
        return self._corr(distances).reshape(-1)

    def gelfand_lower_tails(
        self,
        n_max: int,
        *,
        d: int = 1,
        n_quad: int | None = None,
    ) -> np.ndarray:
        """Return squared tails of a positive discrete covariance.

        Gauss-Legendre nodes represent normalized Lebesgue measure in one
        dimension. In higher dimensions, a deterministic equal-weight Sobol
        measure is used. Either discrete probability measure gives a lower
        bound on the continuum supremum width in exact arithmetic.
        """
        n_max = int(n_max)
        d = int(d)
        if d < 1:
            raise ValueError("d must be positive")
        if n_quad is None:
            n_quad = max(n_max + 64, 400 if d == 1 else 512)
        if not 0 <= n_max < n_quad:
            raise ValueError("n_max must be smaller than n_quad")

        if d == 1:
            nodes, weights = leggauss(n_quad)
            points = torch.from_numpy(nodes).to(self.dtype).reshape(-1, 1)
            sqrt_weights = np.sqrt(0.5 * weights)
        else:
            engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=0)
            points = 2.0 * engine.draw(n_quad).to(self.dtype) - 1.0
            sqrt_weights = np.full(n_quad, 1.0 / math.sqrt(n_quad))

        gram = self.eval(points, points).cpu().numpy()
        covariance = sqrt_weights[:, None] * gram * sqrt_weights[None, :]
        return _discrete_covariance_lower_tails(
            covariance,
            n_max,
            exact_trace=1.0,
        )


class PeriodicSobolevMixedKernel:
    """Periodic mixed-smoothness Sobolev kernel on [0,1]^d."""

    domain = (0.0, 1.0)
    grid_kind = "periodic"

    def __init__(
        self,
        m: int = 1,
        d: int = 1,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        self.m = int(m)
        self.d = int(d)
        self.dtype = dtype
        degree = 2 * self.m
        numbers = bernoulli(degree)
        coefficients = [
            float(comb(degree, j, exact=True) * numbers[j])
            for j in range(degree + 1)
        ]
        self.coeffs = torch.tensor(coefficients, dtype=dtype, device=device)
        self.pref = float((-1.0) ** (self.m - 1) / factorial(degree))
        self._k1_0 = 1.0 + self.pref * float(coefficients[-1])

    def _bern(self, distance: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(distance)
        for coefficient in self.coeffs.to(distance.device):
            result = result * distance + coefficient
        return result

    def _k1(self, distance: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.pref * self._bern(distance)

    def eval(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = X.reshape(X.shape[0], -1).to(self.dtype)
        Y = Y.reshape(Y.shape[0], -1).to(self.dtype)
        difference = X[:, None, :] - Y[None, :, :]
        distance = torch.abs(difference - torch.round(difference))
        return self._k1(distance).prod(dim=2)

    def diagonal(self, X: torch.Tensor) -> torch.Tensor:
        d = X.reshape(X.shape[0], -1).shape[1]
        return torch.full(
            (X.shape[0],), self._k1_0**d, dtype=self.dtype, device=X.device
        )

    def gelfand_lower_tails(self, n_max: int) -> np.ndarray:
        """Return squared complex Fourier widths as lower bounds for real widths."""
        n_max = int(n_max)
        distinct_count = n_max + 2
        values = [1.0] + [
            (2.0 * math.pi * k) ** (-2 * self.m)
            for k in range(1, distinct_count)
        ]
        multiplicities = [1] + [2] * (distinct_count - 1)
        heap = [(-1.0, (0,) * self.d)]
        seen = {(0,) * self.d}
        eigenvalues: list[float] = []
        count = 0
        while heap and count <= n_max:
            negative_value, index = heapq.heappop(heap)
            multiplicity = math.prod(multiplicities[j] for j in index)
            eigenvalues.extend([-negative_value] * multiplicity)
            count += multiplicity
            for coordinate in range(self.d):
                neighbor = (
                    index[:coordinate]
                    + (index[coordinate] + 1,)
                    + index[coordinate + 1 :]
                )
                if neighbor not in seen and neighbor[coordinate] < distinct_count:
                    seen.add(neighbor)
                    heapq.heappush(
                        heap,
                        (
                            negative_value
                            * values[neighbor[coordinate]]
                            / values[index[coordinate]],
                            neighbor,
                        ),
                    )

        trace = self._k1_0**self.d
        removed = np.cumsum(np.asarray(eigenvalues[: n_max + 1]))
        tails = np.concatenate([[trace], trace - removed])[: n_max + 1]
        return np.maximum(tails, 0.0)

    def prepare(self, Xd: torch.Tensor) -> None:
        self._Xd = Xd.reshape(Xd.shape[0], -1).to(self.dtype)
        self._diag = self.diagonal(self._Xd)

    def diag_grid(self) -> torch.Tensor:
        return self._diag

    def col(self, j: int) -> torch.Tensor:
        return self.eval(self._Xd, self._Xd[j : j + 1]).reshape(-1)


class PaleyWienerSincKernel:
    """Band-limited sinc kernel on [-1,1]."""

    domain = (-1.0, 1.0)
    grid_kind = "equidistant"

    def __init__(
        self,
        c: float | None = None,
        n_eff: float | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        if (c is None) == (n_eff is None):
            raise ValueError("give exactly one of c and n_eff")
        self.c = float(c) if c is not None else float(n_eff) * math.pi / 2.0
        self.n_eff = 2.0 * self.c / math.pi
        self.diag_val = self.c / math.pi
        self.dtype = dtype

    def _kfun(self, difference: torch.Tensor) -> torch.Tensor:
        return self.diag_val * torch.sinc(self.c * difference / math.pi)

    def eval(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = X.reshape(-1, 1).to(self.dtype)
        Y = Y.reshape(1, -1).to(self.dtype)
        return self._kfun(X - Y)

    def diagonal(self, X: torch.Tensor) -> torch.Tensor:
        count = X.reshape(-1, 1).shape[0]
        return torch.full(
            (count,), self.diag_val, dtype=self.dtype, device=X.device
        )

    def gelfand_lower_tails(
        self, n_max: int, n_quad: int | None = None
    ) -> np.ndarray:
        """Return safely adjusted squared Gauss-Legendre covariance tails."""
        n_max = int(n_max)
        if n_quad is None:
            n_quad = max(200, int(8 * self.n_eff) + 50, n_max + 1)
        if not 0 <= n_max < n_quad:
            raise ValueError("n_max must be smaller than n_quad")

        nodes, weights = leggauss(n_quad)
        differences = nodes[:, None] - nodes[None, :]
        gram = self.diag_val * np.sinc(self.c * differences / math.pi)
        sqrt_weights = np.sqrt(0.5 * weights)
        covariance = sqrt_weights[:, None] * gram * sqrt_weights[None, :]
        return _discrete_covariance_lower_tails(
            covariance,
            n_max,
            exact_trace=self.diag_val,
        )

    def prepare(self, Xd: torch.Tensor) -> None:
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
        return self._kfun(self._Xd - self._Xd[j : j + 1]).reshape(-1)

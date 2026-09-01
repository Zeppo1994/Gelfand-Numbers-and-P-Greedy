import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from greedy import PGreedy, power_function
from kernels import (
    LegendreMercerKernel,
    MaternKernel,
    PaleyWienerSincKernel,
    PeriodicSobolevMixedKernel,
)
from legendre import (
    comparison_figure as legendre_comparison_figure,
    points_figure as legendre_points_figure,
)
from matern import points_figure as matern_points_figure
from paley_wiener import points_figure as paley_wiener_points_figure
from periodic_mixed import points_figure as periodic_mixed_points_figure
from run_manuscript_figures import STAGES
from lower_bounds import (
    box_grid,
    plot_gelfand_lower_bound,
    plot_sampling_estimate,
    sampling_vs_lower_bound,
    finalize_figure,
)


@pytest.mark.parametrize(
    ("kernel", "domain", "dimension"),
    [
        (LegendreMercerKernel(s=2), (-1.0, 1.0), 1),
        (MaternKernel(nu=1.5), (-1.0, 1.0), 2),
        (PeriodicSobolevMixedKernel(m=2, d=2), (0.0, 1.0), 2),
        (PaleyWienerSincKernel(n_eff=8), (-1.0, 1.0), 1),
    ],
)
def test_kernel_gram_is_symmetric_psd_with_correct_diagonal(
    kernel, domain, dimension
):
    generator = torch.Generator().manual_seed(1)
    lo, hi = domain
    points = lo + (hi - lo) * torch.rand(
        (12, dimension), generator=generator, dtype=kernel.dtype
    )
    gram = kernel.eval(points, points)

    assert torch.allclose(gram, gram.T, atol=2e-12, rtol=2e-12)
    assert torch.linalg.eigvalsh(gram).min() >= -2e-10
    assert torch.allclose(
        torch.diagonal(gram),
        kernel.diagonal(points),
        atol=2e-12,
        rtol=2e-12,
    )


def test_incremental_greedy_matches_direct_power_recomputation():
    kernel = MaternKernel(nu=1.5, ell=0.3)
    grid = box_grid(96, 1, kernel.dtype, kind="equidistant")
    greedy = PGreedy(kernel, max_iter=12, dtype=kernel.dtype).fit(grid)
    direct = torch.stack(
        [
            power_function(kernel, greedy.ctrs_[:m], grid).max()
            for m in range(1, greedy.n_)
        ]
    )

    assert torch.allclose(
        direct,
        greedy.g_curve()[1 : greedy.n_],
        atol=2e-9,
        rtol=2e-9,
    )


def test_paley_wiener_grid_is_equidistant_and_includes_endpoints():
    kernel = PaleyWienerSincKernel(n_eff=8)
    grid = box_grid(
        9,
        1,
        kernel.dtype,
        kind=kernel.grid_kind,
        domain=kernel.domain,
    )
    expected = torch.linspace(*kernel.domain, 9, dtype=kernel.dtype)[:, None]
    assert torch.equal(grid, expected)


def test_shared_equidistant_and_periodic_grids_have_distinct_endpoints():
    dtype = torch.float64
    equidistant = box_grid(5, 1, dtype, kind="equidistant", domain=(0.0, 1.0))
    periodic = box_grid(5, 1, dtype, kind="periodic", domain=(0.0, 1.0))

    assert torch.equal(equidistant, torch.linspace(0.0, 1.0, 5, dtype=dtype)[:, None])
    assert torch.equal(
        periodic, torch.linspace(0.0, 1.0, 6, dtype=dtype)[:-1, None]
    )


def test_legendre_tail_has_normalized_covariance_factor():
    kernel = LegendreMercerKernel(s=2)
    tails = kernel.gelfand_lower_tails(12)
    indices = np.arange(12, dtype=np.float64)
    eigenvalues = 1.0 / (1.0 + (indices * (indices + 1.0)) ** 2)

    assert tails.shape == (13,)
    assert np.allclose(tails[:-1] - tails[1:], 0.5 * eigenvalues)
    assert np.all(np.diff(tails) < 0.0)


@pytest.mark.parametrize("smoothness", [2, 3])
def test_legendre_resolvent_matches_long_mercer_sum(smoothness):
    points = torch.tensor(
        [-0.9, -0.25, 0.2, 0.8],
        dtype=torch.float64,
    )[:, None]
    n_terms = 4096
    values = torch.empty((points.shape[0], n_terms), dtype=points.dtype)
    values[:, 0] = 1.0
    values[:, 1] = points[:, 0]
    for degree in range(1, n_terms - 1):
        values[:, degree + 1] = (
            (2 * degree + 1) * points[:, 0] * values[:, degree]
            - degree * values[:, degree - 1]
        ) / (degree + 1)
    degrees = torch.arange(n_terms, dtype=points.dtype)
    normalized = values * torch.sqrt((2.0 * degrees + 1.0) / 2.0)
    eigenvalues = torch.sigmoid(
        -smoothness * torch.log(degrees * (degrees + 1.0))
    )
    expected = (
        normalized * torch.sqrt(eigenvalues)
    ) @ (
        normalized * torch.sqrt(eigenvalues)
    ).T
    actual = LegendreMercerKernel(s=smoothness).eval(points, points)

    assert torch.allclose(actual, expected, atol=3e-11, rtol=3e-11)


def test_legendre_prepared_resolvent_avoids_feature_table():
    kernel = LegendreMercerKernel(s=2)
    grid = box_grid(33, 1, kernel.dtype, kind="chebyshev")
    kernel.prepare(grid)

    assert kernel._p_plus.numel() == 2 * grid.shape[0]
    assert torch.allclose(
        kernel.col(11),
        kernel.eval(grid, grid[11:12])[:, 0],
        atol=3e-11,
        rtol=3e-11,
    )


@pytest.mark.parametrize(("dimension", "n_quad"), [(1, 48), (3, 64)])
def test_matern_discrete_covariance_tails_are_valid(dimension, n_quad):
    kernel = MaternKernel(nu=1.5, ell=0.5)
    tails = kernel.gelfand_lower_tails(
        12,
        d=dimension,
        n_quad=n_quad,
    )

    assert tails.shape == (13,)
    assert tails[0] == pytest.approx(1.0, rel=2e-13)
    assert np.all(tails >= 0.0)
    assert np.all(np.diff(tails) <= 1e-14)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_periodic_fourier_tails_are_nonnegative_and_decreasing(dimension):
    kernel = PeriodicSobolevMixedKernel(m=1, d=dimension)
    tails = kernel.gelfand_lower_tails(15)

    assert tails.shape == (16,)
    assert tails[0] == pytest.approx(kernel._k1_0**dimension)
    assert np.all(tails >= 0.0)
    assert np.all(np.diff(tails) <= 1e-14)


def test_paley_wiener_covariance_tails_are_nonnegative_and_decreasing():
    kernel = PaleyWienerSincKernel(n_eff=8)
    tails = kernel.gelfand_lower_tails(12, n_quad=48)

    assert tails.shape == (13,)
    assert tails[0] == pytest.approx(kernel.diag_val, rel=2e-14)
    assert np.all(tails >= 0.0)
    assert np.all(np.diff(tails) <= 1e-14)




def test_shared_sampling_pipeline_aligns_sampling_and_lower_bound_data():
    result = sampling_vs_lower_bound(
        MaternKernel(nu=1.5, ell=0.3),
        m_max=8,
        sel_grid=32,
        n_cap=3,
        lower_bound_kwargs={"d": 1, "n_quad": 32},
    )

    assert result["sampling"].shape == result["sampling_n"].shape
    assert result["lower"].shape == result["lower_n"].shape
    assert np.all(result["lower"] >= 0.0)
    assert result["sampling_at_lower_n"].shape == result["lower_n"].shape
    assert result["ratio"].shape == result["lower_n"].shape


def test_shared_plot_styles_and_finalize(tmp_path):
    fig, axis = plt.subplots()
    sampling = plot_sampling_estimate(axis, [1, 2], [1.0, 0.5])
    lower = plot_gelfand_lower_bound(axis, [1, 2], [0.8, 0.3])
    output = tmp_path / "plot.png"
    saved = finalize_figure(fig, output)

    assert sampling.get_marker() == "o"
    assert lower.get_linestyle() == "-"
    assert saved == output
    assert output.exists()


def test_legendre_comparison_uses_lower_bounds_by_default(tmp_path):
    output = tmp_path / "legendre.png"
    result = legendre_comparison_figure(
        max_iter=8,
        sel_grid=32,
        n_cap=4,
        out=output,
    )

    assert output.exists()
    for tag in ("s2", "s3"):
        assert result[f"{tag}_sampling"].shape == result[f"{tag}_sampling_n"].shape
        assert result[f"{tag}_lower"].shape == result[f"{tag}_lower_n"].shape
        assert result[f"{tag}_n_used"] == 8


@pytest.mark.parametrize(
    ("driver", "kwargs", "filename"),
    [
        pytest.param(
            legendre_points_figure,
            {"smoothness_values": (2,), "m": 4, "grid": 32},
            "legendre_points.png",
            id="legendre",
        ),
        pytest.param(
            matern_points_figure,
            {"n_points": 8, "snapshot_size": 3, "grid": 64, "query_grid": 64},
            "matern_points.png",
            id="matern",
        ),
        pytest.param(
            periodic_mixed_points_figure,
            {"counts": (4, 8), "sel_grid": 64},
            "periodic_mixed_points.png",
            id="periodic-mixed",
        ),
        pytest.param(
            paley_wiener_points_figure,
            {"n_effs": (4, 8), "grid": 128},
            "paley_wiener_points.png",
            id="paley-wiener",
        ),
    ],
)
def test_point_figure_drivers_create_manuscript_outputs(
    tmp_path, driver, kwargs, filename
):
    output = tmp_path / filename
    result = driver(out=output, **kwargs)

    assert output.exists()
    assert any(key == "centers" or key.endswith("_centers") for key in result)


def test_publication_runner_contains_comparison_and_point_stages():
    assert tuple(name for name, _, _ in STAGES) == (
        "legendre",
        "legendre_points",
        "matern",
        "matern_points",
        "periodic_mixed",
        "periodic_mixed_points",
        "paley_wiener",
        "paley_wiener_points",
    )

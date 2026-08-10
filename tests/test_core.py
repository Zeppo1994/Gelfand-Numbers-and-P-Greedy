from __future__ import annotations

import numpy as np
import inspect
import pytest
import torch
import matplotlib.pyplot as plt

from legendre import rates_figure as legendre_rates_figure
from periodic_mixed import rates_figure as periodic_rates_figure
from greedy import PGreedy, power_function
from kernels import (
    LegendreMercerKernel,
    MaternKernel,
    PaleyWienerSincKernel,
    PeriodicSobolevMixedKernel,
)
from widths import gelfand_widths, plot_gelfand_bounds, sampling_vs_gelfand

@pytest.mark.parametrize("rates_figure", [legendre_rates_figure, periodic_rates_figure])
def test_diagnostic_overlays_are_opt_in(rates_figure):
    parameter = inspect.signature(rates_figure).parameters["diagnostic_overlays"]
    assert parameter.default is False





@pytest.mark.parametrize(
    ("kernel", "domain", "dimension"),
    [
        (LegendreMercerKernel(n_trunc=128), (-1.0, 1.0), 1),
        (MaternKernel(nu=1.5), (-1.0, 1.0), 2),
        (PeriodicSobolevMixedKernel(m=2, d=2), (0.0, 1.0), 2),
        (PaleyWienerSincKernel(n_eff=8), (-1.0, 1.0), 1),
    ],
)
def test_kernel_gram_is_symmetric_psd_with_correct_diagonal(kernel, domain, dimension):
    x = torch.linspace(*domain, 20, dtype=torch.float64)
    X = torch.stack((x, torch.roll(x, 3)), dim=1) if dimension == 2 else x[:, None]
    gram = kernel.eval(X, X)
    eig = torch.linalg.eigvalsh(0.5 * (gram + gram.T))

    assert torch.allclose(gram, gram.T, atol=1e-12, rtol=0)
    assert float(eig.min()) >= -1e-12
    assert torch.allclose(torch.diag(gram), kernel.diagonal(X), atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    ("kernel", "dimension", "domain"),
    [
        (MaternKernel(nu=1.5), 2, (-0.8, 0.8)),
        (PeriodicSobolevMixedKernel(m=2, d=2), 2, (0.1, 0.9)),
        (PaleyWienerSincKernel(n_eff=8), 1, (-0.8, 0.8)),
    ],
)
def test_analytic_kernel_gradient_matches_finite_difference(kernel, dimension, domain):
    generator = torch.Generator().manual_seed(2)
    X = domain[0] + (domain[1] - domain[0]) * torch.rand(
        4, dimension, dtype=torch.float64, generator=generator
    )
    Y = domain[0] + (domain[1] - domain[0]) * torch.rand(
        6, dimension, dtype=torch.float64, generator=generator
    )
    h = 1e-6
    numeric = []
    for coordinate in range(dimension):
        direction = torch.zeros(dimension, dtype=torch.float64)
        direction[coordinate] = h
        numeric.append((kernel.eval(X + direction, Y) - kernel.eval(X - direction, Y)) / (2 * h))
    numeric = torch.stack(numeric, dim=-1)

    assert torch.allclose(kernel.eval_grad(X, Y), numeric, atol=2e-8, rtol=2e-8)


def test_incremental_greedy_matches_direct_power_recomputation():
    X = torch.linspace(-1, 1, 97, dtype=torch.float64)[:, None]
    kernel = MaternKernel(nu=1.5, ell=0.3)
    greedy = PGreedy(kernel, max_iter=18).fit(X)
    direct = torch.stack(
        [power_function(kernel, greedy.ctrs_[:m], X).max() for m in range(1, greedy.n_)]
    )

    assert greedy.idx_.unique().numel() == greedy.n_
    assert torch.all(greedy.g_curve()[1:] <= greedy.g_curve()[:-1])
    assert torch.allclose(direct, greedy.g_curve()[1 : greedy.n_], atol=2e-9, rtol=2e-9)


def test_truncated_dual_lower_bound_is_conservative():
    count = 128
    X = torch.arange(count, dtype=torch.float64)[:, None] / count
    kernel = PeriodicSobolevMixedKernel(m=1, d=1)
    indices = np.array([1, 3, 5, 9])
    exact_at_closed_shells = np.sqrt(
        np.clip(kernel.gelfand_tails(int(indices[-1])), 0.0, None)
    )[indices]

    _, lower, info = gelfand_widths(
        kernel,
        X,
        indices,
        n_iter=6,
        compress_irls=True,
        refine=False,
        return_info=True,
    )
    _, lower_full = gelfand_widths(
        kernel,
        X,
        indices,
        n_iter=6,
        compress_irls=False,
        refine=False,
    )

    assert not info["upper_certified"].any()
    assert np.all(lower.numpy() <= exact_at_closed_shells + 1e-10)
    assert np.all(lower.numpy() <= lower_full.numpy() + 1e-10)


def test_sampling_result_exposes_estimate_and_certification():
    result = sampling_vs_gelfand(
        MaternKernel(nu=1.5, ell=0.3),
        m_max=8,
        sel_grid=32,
        cn_grid=20,
        n_cn=3,
        n_cap=3,
        exchange=False,
        certify_tol=0.2,
        certify_evals=2_000,
    )

    assert result["g"].shape == result["m_list"].shape
    assert result["cn_certified"].dtype == np.bool_
    assert result["cn_certified"].shape == result["cn"].shape
    assert np.all(result["cn_lo"] <= result["cn"] + 1e-12)


def test_gelfand_plot_distinguishes_certified_and_estimated_upper_values():
    n = np.array([1, 2, 4, 8])
    lower = np.array([0.8, 0.5, 0.3, 0.2])
    upper = np.array([1.0, 0.7, 0.45, 0.32])
    certified = np.array([True, True, False, False])
    fig, ax = plt.subplots()

    artists = plot_gelfand_bounds(
        ax, n, lower, upper, certified, series_label=r"$m=2$"
    )

    assert artists["lower"].get_linestyle() == "-"
    assert artists["certified_upper"].get_linestyle() == "-"
    assert artists["estimated_upper"].get_linestyle() == "--"
    assert artists["certified_fill"] is not None
    assert artists["estimated_fill"] is not None
    labels = [collection.get_label() for collection in ax.collections]
    assert any("certified" in label for label in labels)
    assert any("estimated" in label for label in labels)
    plt.close(fig)

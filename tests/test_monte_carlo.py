import numpy as np
import pytest

from stochastic_inference_engine import MonteCarloIntegrator


def test_constant_integral():
    estimate, se = MonteCarloIntegrator.integrate_1d(
        lambda x: np.full_like(x, 3.0),
        a=1.0,
        b=5.0,
        n_samples=1000,
        seed=42,
    )

    assert estimate == pytest.approx(12.0, abs=1e-12)
    assert se == pytest.approx(0.0, abs=1e-12)


def test_standard_monte_carlo_integral():
    # Integral of x^2 from 0 to 1 = 1/3.
    estimate, se = MonteCarloIntegrator.integrate_1d(
        lambda x: x**2,
        a=0.0,
        b=1.0,
        n_samples=50_000,
        seed=42,
    )

    assert abs(estimate - 1 / 3) < 5 * se


def test_standard_reproducibility():
    kwargs = dict(
        func=lambda x: np.sin(x),
        a=0.0,
        b=np.pi,
        n_samples=1000,
        seed=42,
    )

    first = MonteCarloIntegrator.integrate_1d(**kwargs)
    second = MonteCarloIntegrator.integrate_1d(**kwargs)

    assert first == second


def test_antithetic_linear_exactness():
    estimate, se = (
        MonteCarloIntegrator.integrate_antithetic(
            lambda x: x,
            a=0.0,
            b=1.0,
            n_samples=1000,
            seed=42,
        )
    )

    assert estimate == pytest.approx(0.5, abs=1e-12)
    assert se == pytest.approx(0.0, abs=1e-12)


def test_antithetic_variance_reduction():
    n_samples = 10_000

    standard_estimate, standard_se = (
        MonteCarloIntegrator.integrate_1d(
            lambda x: x,
            0.0, 1.0, n_samples, seed=42
        )
    )

    anti_estimate, anti_se = (
        MonteCarloIntegrator.integrate_antithetic(
            lambda x: x,
            0.0, 1.0, n_samples, seed=42
        )
    )

    assert anti_se < standard_se
    assert anti_estimate == pytest.approx(0.5, abs=1e-12)


def test_antithetic_quadratic_integral():
    estimate, se = (
        MonteCarloIntegrator.integrate_antithetic(
            lambda x: x**2,
            0.0, 1.0, 20_000, seed=42
        )
    )

    assert abs(estimate - 1 / 3) < 5 * se


def test_importance_sampling_exact():
    rng = np.random.default_rng(42)

    estimate, se = MonteCarloIntegrator.importance_sampling(
        target_func=lambda x: x**2,
        proposal_sample_func=lambda n: rng.uniform(
            0.0, 1.0, size=n
        ),
        proposal_pdf_func=lambda x: np.ones_like(x),
        n_samples=50_000,
    )

    assert abs(estimate - 1 / 3) < 5 * se


def test_importance_sampling_normal_integral():
    rng = np.random.default_rng(42)

    def normal_pdf(x):
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    estimate, se = MonteCarloIntegrator.importance_sampling(
        target_func=normal_pdf,
        proposal_sample_func=lambda n: rng.normal(size=n),
        proposal_pdf_func=normal_pdf,
        n_samples=10_000,
    )

    assert estimate == pytest.approx(1.0, abs=1e-12)
    assert se == pytest.approx(0.0, abs=1e-12)


def test_target_expectation():
    rng = np.random.default_rng(42)

    def normal_pdf(x):
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    estimate, se = (
        MonteCarloIntegrator.expectation_importance_sampling(
            func=lambda x: x**2,
            target_pdf_func=normal_pdf,
            proposal_sample_func=lambda n: rng.normal(size=n),
            proposal_pdf_func=normal_pdf,
            n_samples=50_000,
        )
    )

    assert abs(estimate - 1.0) < 5 * se


def test_invalid_sample_counts():
    with pytest.raises(ValueError):
        MonteCarloIntegrator.integrate_1d(
            lambda x: x, 0.0, 1.0, n_samples=1
        )

    with pytest.raises(ValueError):
        MonteCarloIntegrator.integrate_antithetic(
            lambda x: x, 0.0, 1.0, n_samples=999
        )


def test_invalid_bounds():
    with pytest.raises(ValueError):
        MonteCarloIntegrator.integrate_1d(
            lambda x: x, 2.0, 1.0, seed=42
        )


def test_invalid_proposal_density():
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        MonteCarloIntegrator.importance_sampling(
            target_func=lambda x: x**2,
            proposal_sample_func=lambda n: rng.uniform(
                size=n
            ),
            proposal_pdf_func=lambda x: np.zeros_like(x),
            n_samples=100,
        )

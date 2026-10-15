import itertools

import numpy as np
import pytest

from stochastic_inference_engine.samplers.mcmc import (
    Ising1DSampler,
    RandomWalkMetropolis2D,
    MCMCDiagnostics,
)


def exact_ising_mean_magnetization(num_spins, beta):
    """Enumerate the exact finite-state Ising distribution."""
    model = Ising1DSampler(num_spins)

    states = np.array(
        list(itertools.product((-1, 1), repeat=num_spins))
    )

    energies = np.array([
        model.energy(state) for state in states
    ])

    weights = np.exp(-beta * energies)
    probabilities = weights / weights.sum()

    magnetizations = states.mean(axis=1)

    return np.sum(probabilities * magnetizations)


def test_ising_energy():
    model = Ising1DSampler(num_spins=3)

    # H = -(1*1 + 1*(-1)) - (1+1-1) = -1
    assert model.energy([1, 1, -1]) == -1.0

    # All +1: H = -2 - 3 = -5
    assert model.energy([1, 1, 1]) == -5.0


def test_ising_shapes_and_spin_values():
    model = Ising1DSampler(num_spins=10)

    for method in (model.run_metropolis, model.run_gibbs):
        states, magnetization = method(
            steps=100, beta=0.5, seed=42
        )

        assert states.shape == (101, 10)
        assert magnetization.shape == (101,)
        assert np.all(np.isin(states, (-1, 1)))

        np.testing.assert_allclose(
            magnetization,
            states.mean(axis=1),
        )


def test_ising_reproducibility():
    model = Ising1DSampler(num_spins=10)

    for method in (model.run_metropolis, model.run_gibbs):
        first = method(100, 0.5, seed=42)
        second = method(100, 0.5, seed=42)

        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("method_name", [
    "run_metropolis",
    "run_gibbs",
])
def test_ising_against_exact_distribution(method_name):
    num_spins = 4
    beta = 0.5

    theoretical = exact_ising_mean_magnetization(
        num_spins, beta
    )

    model = Ising1DSampler(num_spins)
    method = getattr(model, method_name)

    _, magnetization = method(
        steps=100_000,
        beta=beta,
        seed=42,
    )

    # Discard initial transient.
    retained = magnetization[10_000:]

    empirical = retained.mean()

    # Conservative tolerance for correlated MCMC samples.
    assert abs(empirical - theoretical) < 0.08


def test_ising_zero_beta():
    # At beta=0, every configuration is equally likely.
    # Therefore the stationary mean magnetization is zero.
    model = Ising1DSampler(num_spins=4)

    _, magnetization = model.run_gibbs(
        steps=50_000,
        beta=0.0,
        seed=123,
    )

    assert abs(magnetization[5000:].mean()) < 0.08

def test_autocovariance():
    x = np.array([1.0, 2.0, 3.0, 4.0])

    result = Ising1DSampler.compute_autocovariance(
        x, max_lag=3
    )

    centered = x - x.mean()

    expected = np.array([
        np.dot(centered, centered) / 4,
        np.dot(centered[:-1], centered[1:]) / 4,
        np.dot(centered[:-2], centered[2:]) / 4,
    ])

    np.testing.assert_allclose(result, expected)


def test_autocovariance_invalid_lag():
    with pytest.raises(ValueError):
        Ising1DSampler.compute_autocovariance(
            np.array([1.0, 2.0, 3.0]),
            max_lag=5,
        )


def standard_normal_logpdf(x, y):
    # Normalization constant is unnecessary for MCMC.
    return -0.5 * (x**2 + y**2)


def test_metropolis_2d_reproducibility():
    sampler = RandomWalkMetropolis2D(
        target_logpdf=standard_normal_logpdf
    )

    first = sampler.sample(
        100, 1.0, seed=42
    )
    second = sampler.sample(
        100, 1.0, seed=42
    )

    np.testing.assert_array_equal(first, second)


def test_metropolis_2d_gaussian_moments():
    sampler = RandomWalkMetropolis2D(
        target_logpdf=standard_normal_logpdf
    )

    samples = sampler.sample(
        steps=60_000,
        step_length=1.0,
        seed=42,
    )

    retained = samples[10_000:]

    means = retained.mean(axis=0)
    covariance = np.cov(retained.T)

    np.testing.assert_allclose(
        means,
        [0.0, 0.0],
        atol=0.12,
    )

    np.testing.assert_allclose(
        covariance,
        np.eye(2),
        atol=0.15,
    )


def test_metropolis_2d_pdf_interface():
    sampler = RandomWalkMetropolis2D(
        target_pdf=lambda x, y: np.exp(
            -0.5 * (x**2 + y**2)
        )
    )

    samples = sampler.sample(
        steps=100,
        step_length=1.0,
        seed=42,
    )

    assert samples.shape == (101, 2)
    assert np.all(np.isfinite(samples))


def test_metropolis_rejects_invalid_initial_state():
    sampler = RandomWalkMetropolis2D(
        target_logpdf=lambda x, y: (
            0.0 if x > 0 else -np.inf
        )
    )

    with pytest.raises(ValueError):
        sampler.sample(
            steps=100,
            step_length=1.0,
            initial_state=(-1.0, 0.0),
            seed=42,
        )


def test_autocorrelation_and_ess():
    rng = np.random.default_rng(42)
    independent = rng.standard_normal(10_000)

    rho = MCMCDiagnostics.autocorrelation(
        independent, max_lag=50
    )

    ess = MCMCDiagnostics.effective_sample_size(
        independent, max_lag=50
    )

    assert rho[0] == pytest.approx(1.0)
    assert abs(rho[1]) < 0.05

    # Independent draws should have ESS near sample count.
    assert 7000 < ess <= 10_000


def test_ess_detects_autocorrelation():
    rng = np.random.default_rng(42)

    n = 10_000
    x = np.zeros(n)

    # Stationary AR(1) with correlation coefficient 0.9.
    phi = 0.9
    noise_scale = np.sqrt(1 - phi**2)

    x[0] = rng.standard_normal()

    for i in range(1, n):
        x[i] = (
            phi * x[i - 1]
            + noise_scale * rng.standard_normal()
        )

    ess = MCMCDiagnostics.effective_sample_size(
        x, max_lag=200
    )

    assert ess < n / 4


def test_invalid_sampler_arguments():
    with pytest.raises(ValueError):
        Ising1DSampler(num_spins=0)

    model = Ising1DSampler(num_spins=5)

    with pytest.raises(ValueError):
        model.run_metropolis(100, beta=-1.0)

    with pytest.raises(ValueError):
        RandomWalkMetropolis2D()

    sampler = RandomWalkMetropolis2D(
        target_logpdf=standard_normal_logpdf
    )

    with pytest.raises(ValueError):
        sampler.sample(100, step_length=0.0)

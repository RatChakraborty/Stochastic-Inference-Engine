
import numpy as np
import pytest

from stochastic_inference_engine.dynamics.sdes import (
    MertonJumpDiffusion,
    MertonAssetPriceModel,
)


def make_additive_model(
    jump_lambda=2.0,
    jump_mu=0.5,
    jump_sigma=0.3,
):
    return MertonJumpDiffusion(
        mu_func=lambda x, t: 0.0,
        sigma_func=lambda x, t: 0.0,
        jump_lambda=jump_lambda,
        jump_mu=jump_mu,
        jump_sigma=jump_sigma,
    )


def test_additive_shape():
    model = make_additive_model()

    result = model.simulate(
        x0=0.0,
        horizon=1.0,
        steps=100,
        paths=20,
        seed=42,
    )

    assert result.shape == (101, 20)
    assert np.all(result[0] == 0.0)


def test_additive_reproducibility():
    model = make_additive_model()

    a = model.simulate(0.0, 1.0, 100, 20, seed=42)
    b = model.simulate(0.0, 1.0, 100, 20, seed=42)

    np.testing.assert_array_equal(a, b)


def test_additive_zero_jumps():
    model = make_additive_model(jump_lambda=0.0)

    result = model.simulate(
        x0=5.0,
        horizon=1.0,
        steps=100,
        paths=100,
        seed=42,
    )

    np.testing.assert_array_equal(
        result, np.full_like(result, 5.0)
    )


def test_additive_jump_moments():
    jump_lambda = 3.0
    jump_mu = 0.5
    jump_sigma = 0.4
    T = 1.0

    model = make_additive_model(
        jump_lambda, jump_mu, jump_sigma
    )

    result = model.simulate(
        x0=0.0,
        horizon=T,
        steps=1,
        paths=100_000,
        seed=42,
    )

    terminal = result[-1]

    theoretical_mean = jump_lambda * T * jump_mu

    theoretical_variance = jump_lambda * T * (
        jump_sigma**2 + jump_mu**2
    )

    mean_se = np.sqrt(
        theoretical_variance / len(terminal)
    )

    assert abs(
        terminal.mean() - theoretical_mean
    ) < 5 * mean_se

    # Broad variance tolerance for a compound-Poisson law
    relative_variance_error = abs(
        terminal.var(ddof=1) - theoretical_variance
    ) / theoretical_variance

    assert relative_variance_error < 0.04


def test_multiple_jumps_per_interval():
    model = make_additive_model(
        jump_lambda=5.0,
        jump_mu=1.0,
        jump_sigma=0.0,
    )

    result = model.simulate(
        x0=0.0,
        horizon=1.0,
        steps=1,
        paths=1000,
        seed=42,
    )

    terminal = result[-1]

    assert np.any(terminal > 1.0)


    assert np.all(terminal >= 0)
    np.testing.assert_allclose(
        terminal, np.round(terminal)
    )


def test_invalid_additive_parameters():
    with pytest.raises(ValueError):
        make_additive_model(jump_lambda=-1.0)

    with pytest.raises(ValueError):
        make_additive_model(jump_sigma=-0.1)

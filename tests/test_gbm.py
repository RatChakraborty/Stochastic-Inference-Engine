import numpy as np
import pytest

from stochastic_inference_engine import GeometricBrownianMotion


def test_gbm_shape_and_initial_value():
    model = GeometricBrownianMotion(
        x0=100.0, mu=0.05, sigma=0.2
    )

    paths = model.simulate(
        horizon=1.0, steps=252, paths=100, seed=42
    )

    assert paths.shape == (253, 100)
    assert np.all(paths[0] == 100.0)


def test_gbm_reproducibility():
    model = GeometricBrownianMotion(
        x0=100.0, mu=0.05, sigma=0.2
    )

    first = model.simulate(1.0, 252, 100, seed=42)
    second = model.simulate(1.0, 252, 100, seed=42)

    np.testing.assert_array_equal(first, second)


def test_gbm_expected_value():
    model = GeometricBrownianMotion(
        x0=100.0, mu=0.05, sigma=0.2
    )

    T = 1.0
    paths = model.simulate(
        horizon=T, steps=252, paths=100_000, seed=42
    )

    terminal = paths[-1]

    theoretical_mean = model.x0 * np.exp(model.mu * T)

    theoretical_variance = (
        model.x0**2
        * np.exp(2 * model.mu * T)
        * np.expm1(model.sigma**2 * T)
    )

    standard_error = np.sqrt(
        theoretical_variance / len(terminal)
    )

    assert abs(terminal.mean() - theoretical_mean) < (
        5 * standard_error
    )


def test_gbm_log_return_moments():
    model = GeometricBrownianMotion(
        x0=100.0, mu=0.05, sigma=0.2
    )

    T = 1.0
    paths = model.simulate(
        horizon=T, steps=252, paths=100_000, seed=123
    )

    log_returns = np.log(paths[-1] / model.x0)

    expected_mean = (model.mu - 0.5 * model.sigma**2) * T
    expected_variance = model.sigma**2 * T

    n = len(log_returns)

    mean_se = np.sqrt(expected_variance / n)
    variance_se = expected_variance * np.sqrt(2 / (n - 1))

    assert abs(log_returns.mean() - expected_mean) < (
        5 * mean_se
    )

    assert abs(
        log_returns.var(ddof=1) - expected_variance
    ) < 5 * variance_se


def test_gbm_positivity():
    model = GeometricBrownianMotion(
        x0=100.0, mu=0.05, sigma=0.2
    )

    paths = model.simulate(
        horizon=1.0, steps=252, paths=1000, seed=42
    )

    assert np.all(paths > 0)


def test_gbm_zero_volatility():
    model = GeometricBrownianMotion(
        x0=100.0, mu=0.05, sigma=0.0
    )

    paths = model.simulate(
        horizon=1.0, steps=100, paths=10, seed=42
    )

    times = np.linspace(0, 1, 101)
    expected = 100.0 * np.exp(0.05 * times)

    np.testing.assert_allclose(
        paths[:, 0], expected, rtol=1e-12
    )


@pytest.mark.parametrize(
    "x0,mu,sigma",
    [
        (0.0, 0.05, 0.2),
        (-100.0, 0.05, 0.2),
        (100.0, 0.05, -0.2),
        (np.nan, 0.05, 0.2),
        (100.0, np.inf, 0.2),
    ],
)
def test_gbm_invalid_parameters(x0, mu, sigma):
    with pytest.raises(ValueError):
        GeometricBrownianMotion(x0, mu, sigma)


@pytest.mark.parametrize(
    "horizon,steps,paths",
    [
        (0.0, 252, 100),
        (-1.0, 252, 100),
        (1.0, 0, 100),
        (1.0, 252, 0),
        (1.0, 2.5, 100),
    ],
)
def test_gbm_invalid_simulation_arguments(
    horizon, steps, paths
):
    model = GeometricBrownianMotion(100.0, 0.05, 0.2)

    with pytest.raises(ValueError):
        model.simulate(horizon, steps, paths, seed=42)

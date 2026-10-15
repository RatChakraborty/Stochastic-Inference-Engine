import numpy as np
import pytest

from stochastic_inference_engine import (
    GeometricBrownianMotion,
    MertonAssetPriceModel,
    SimulationConfig,
    SyntheticDataGenerator,
)


def make_gbm():
    return GeometricBrownianMotion(
        x0=100.0,
        mu=0.08,
        sigma=0.2,
    )


def make_config(paths=1000, steps=252, seed=42):
    return SimulationConfig(
        horizon=1.0,
        steps=steps,
        paths=paths,
        seed=seed,
    )

def test_synthetic_shapes():
    dataset = SyntheticDataGenerator.generate(
        make_gbm(), make_config(paths=50, steps=100)
    )

    assert dataset.prices.shape == (101, 50)
    assert dataset.log_returns.shape == (100, 50)
    assert dataset.simple_returns.shape == (100, 50)
    assert dataset.time_grid.shape == (101,)

def test_synthetic_reproducibility():
    model = make_gbm()
    config = make_config()

    first = SyntheticDataGenerator.generate(model, config)
    second = SyntheticDataGenerator.generate(model, config)

    np.testing.assert_array_equal(
        first.prices, second.prices
    )

    np.testing.assert_array_equal(
        first.log_returns, second.log_returns
    )

def test_return_identities():
    dataset = SyntheticDataGenerator.generate(
        make_gbm(), make_config()
    )

    np.testing.assert_allclose(
        dataset.simple_returns,
        np.expm1(dataset.log_returns),
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        dataset.log_returns,
        np.log1p(dataset.simple_returns),
        rtol=1e-10,
        atol=1e-12,
    )

def test_positive_prices():
    dataset = SyntheticDataGenerator.generate(
        make_gbm(), make_config()
    )

    assert np.all(dataset.prices > 0)

def test_gbm_log_return_mean():
    model = make_gbm()
    config = make_config(paths=20_000, steps=10)

    dataset = SyntheticDataGenerator.generate(
        model, config
    )

    expected_mean = (
        model.mu - 0.5 * model.sigma**2
    ) * config.dt

    expected_variance = model.sigma**2 * config.dt

    n = dataset.log_returns.size
    standard_error = np.sqrt(expected_variance / n)

    assert abs(
        dataset.mean_log_return() - expected_mean
    ) < 5 * standard_error

def test_gbm_log_return_variance():
    model = make_gbm()
    config = make_config(paths=20_000, steps=10)

    dataset = SyntheticDataGenerator.generate(
        model, config
    )

    expected_variance = model.sigma**2 * config.dt
    empirical_variance = dataset.variance_log_return()

    relative_error = abs(
        empirical_variance - expected_variance
    ) / expected_variance

    assert relative_error < 0.02

def test_gbm_terminal_mean():
    model = make_gbm()
    config = make_config(paths=50_000, steps=1)

    dataset = SyntheticDataGenerator.generate(
        model, config
    )

    expected = model.x0 * np.exp(
        model.mu * config.horizon
    )

    terminal = dataset.terminal_prices
    standard_error = (
        terminal.std(ddof=1) / np.sqrt(len(terminal))
    )

    assert abs(terminal.mean() - expected) < 5 * standard_error


def test_merton_log_return_moments():
    model = MertonAssetPriceModel(
        x0=100.0,
        mu=0.08,
        sigma=0.2,
        jump_lambda=2.0,
        jump_mu=-0.05,
        jump_sigma=0.1,
    )

    config = make_config(paths=100_000, steps=1)

    dataset = SyntheticDataGenerator.generate(
        model, config
    )

    theoretical_mean, theoretical_variance = (
        model.log_return_moments(config.horizon)
    )

    empirical = dataset.log_returns.ravel()

    mean_se = np.sqrt(
        theoretical_variance / len(empirical)
    )

    assert abs(
        empirical.mean() - theoretical_mean
    ) < 5 * mean_se

    relative_variance_error = abs(
        empirical.var(ddof=1) - theoretical_variance
    ) / theoretical_variance

    assert relative_variance_error < 0.04

def test_terminal_summary():
    dataset = SyntheticDataGenerator.generate(
        make_gbm(), make_config()
    )

    summary = dataset.terminal_summary()

    assert summary["min"] <= summary["median"]
    assert summary["median"] <= summary["max"]
    assert summary["q05"] <= summary["q95"]
    assert summary["std"] >= 0


def test_synthetic_metadata():
    dataset = SyntheticDataGenerator.generate(
        make_gbm(), make_config()
    )

    assert dataset.model_name == "GeometricBrownianMotion"
    assert dataset.model_parameters["x0"] == 100.0
    assert dataset.model_parameters["mu"] == 0.08
    assert dataset.model_parameters["sigma"] == 0.2

def test_unsupported_model():
    class FakeModel:
        pass

    with pytest.raises(TypeError):
        SyntheticDataGenerator.generate(
            FakeModel(), make_config()
        )
def test_gbm_annualized_volatility():
    model = make_gbm()
    config = make_config(
        paths=1000, steps=252
    )

    dataset = SyntheticDataGenerator.generate(
        model, config
    )

    empirical = dataset.annualized_volatility(
        periods_per_year=252
    )

    assert abs(empirical - model.sigma) < 0.01


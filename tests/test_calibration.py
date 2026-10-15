import numpy as np
import pytest

from stochastic_inference_engine import (
    GBMCalibrator,
    GeometricBrownianMotion,
    SimulationConfig,
    SyntheticDataGenerator,
)

def test_gbm_mle_formulas():
    returns = np.array([
        0.01, -0.02, 0.03, 0.005, -0.01
    ])

    dt = 1 / 252

    result = GBMCalibrator.fit(returns, dt)

    expected_variance = np.var(returns, ddof=0)

    expected_sigma = np.sqrt(
        expected_variance / dt
    )

    expected_mu = (
        np.mean(returns) / dt
        + 0.5 * expected_sigma**2
    )

    assert result.parameters["sigma"] == pytest.approx(
        expected_sigma
    )

    assert result.parameters["mu"] == pytest.approx(
        expected_mu
    )

def test_gbm_parameter_recovery():
    true_mu = 0.08
    true_sigma = 0.20

    model = GeometricBrownianMotion(
        x0=100.0,
        mu=true_mu,
        sigma=true_sigma,
    )

    config = SimulationConfig(
        horizon=1.0,
        steps=252,
        paths=2000,
        seed=42,
    )

    dataset = SyntheticDataGenerator.generate(
        model, config
    )

    result = GBMCalibrator.fit(
        dataset.log_returns,
        dt=dataset.dt,
    )

    assert abs(
        result.parameters["mu"] - true_mu
    ) < 5 * result.standard_errors["mu"]

    assert abs(
        result.parameters["sigma"] - true_sigma
    ) < 5 * result.standard_errors["sigma"]

def test_gbm_2d_returns():
    returns = np.array([
        [0.01, 0.02],
        [-0.01, 0.03],
        [0.005, -0.02],
    ])

    result = GBMCalibrator.fit(
        returns,
        dt=1 / 252,
    )

    assert result.n_observations == 6
    assert result.converged

def test_gbm_mle_maximizes_likelihood():
    rng = np.random.default_rng(42)

    returns = rng.normal(
        loc=0.0002,
        scale=0.01,
        size=5000,
    )

    dt = 1 / 252

    result = GBMCalibrator.fit(returns, dt)

    fitted_ll = result.log_likelihood

    alternative_ll = GBMCalibrator.log_likelihood(
        returns,
        dt=dt,
        mu=0.5,
        sigma=0.4,
    )

    assert fitted_ll >= alternative_ll

def test_gbm_calibration_reproducibility():
    returns = np.array([
        0.01, -0.02, 0.005, 0.015
    ])

    first = GBMCalibrator.fit(returns, 1 / 252)
    second = GBMCalibrator.fit(returns, 1 / 252)

    assert first.parameters == second.parameters
    assert first.log_likelihood == second.log_likelihood

@pytest.mark.parametrize(
    "dt",
    [0.0, -1.0, np.nan, np.inf],
)
def test_invalid_dt(dt):
    with pytest.raises(ValueError):
        GBMCalibrator.fit(
            [0.01, -0.02, 0.03],
            dt,
        )

def test_invalid_returns():
    with pytest.raises(ValueError):
        GBMCalibrator.fit([0.01], 1 / 252)

    with pytest.raises(ValueError):
        GBMCalibrator.fit(
            [0.01, np.nan, 0.02],
            1 / 252,
        )

    with pytest.raises(ValueError):
        GBMCalibrator.fit(
            [0.01, np.inf, 0.02],
            1 / 252,
        )

def test_constant_returns():
    with pytest.raises(ValueError):
        GBMCalibrator.fit(
            [0.01, 0.01, 0.01],
            1 / 252,
        )

def test_information_criteria():
    returns = np.array([
        0.01, -0.02, 0.03, 0.005
    ])

    result = GBMCalibrator.fit(
        returns,
        dt=1 / 252,
    )

    assert result.n_parameters == 2

    assert result.aic == pytest.approx(
        4 - 2 * result.log_likelihood
    )

    assert result.bic == pytest.approx(
        2 * np.log(4)
        - 2 * result.log_likelihood
    )

def test_standard_errors():
    rng = np.random.default_rng(7)

    returns = rng.normal(
        loc=0.001,
        scale=0.02,
        size=1000,
    )

    result = GBMCalibrator.fit(
        returns,
        dt=1 / 252,
    )

    assert result.standard_errors["mu"] > 0
    assert result.standard_errors["sigma"] > 0

def test_log_likelihood_consistency():
    returns = np.array([
        0.01, -0.01, 0.02, 0.005
    ])

    result = GBMCalibrator.fit(
        returns,
        dt=1 / 252,
    )

    direct = GBMCalibrator.log_likelihood(
        returns,
        dt=1 / 252,
        mu=result.parameters["mu"],
        sigma=result.parameters["sigma"],
    )

    assert result.log_likelihood == pytest.approx(
        direct
    )

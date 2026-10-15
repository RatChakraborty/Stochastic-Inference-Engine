import numpy as np
import pytest

from scipy.stats import norm, poisson
from scipy.special import logsumexp

from stochastic_inference_engine import (
    GBMCalibrator,
    MertonCalibrator,
    MertonAssetPriceModel,
    SimulationConfig,
    SyntheticDataGenerator,
)


def make_merton():
    return MertonAssetPriceModel(
        x0=100.0,
        mu=0.08,
        sigma=0.20,
        jump_lambda=4.0,
        jump_mu=-0.12,
        jump_sigma=0.12,
    )

def test_zero_jump_likelihood_matches_gbm():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, size=100)

    gbm_ll = GBMCalibrator.log_likelihood(
        returns,
        dt=1 / 252,
        mu=0.08,
        sigma=0.20,
    )

    merton_ll = MertonCalibrator.log_likelihood(
        returns,
        dt=1 / 252,
        mu=0.08,
        sigma=0.20,
        jump_lambda=0.0,
        jump_mu=-0.1,
        jump_sigma=0.1,
    )

    assert merton_ll == pytest.approx(
        gbm_ll, abs=1e-10
    )

def test_merton_likelihood_direct_formula():
    returns = np.array([-0.12, 0.01, 0.08])

    dt = 0.25
    mu = 0.08
    sigma = 0.2
    lam = 2.0
    jump_mu = -0.1
    jump_sigma = 0.15

    calculated = MertonCalibrator.log_likelihood(
        returns,
        dt,
        mu,
        sigma,
        lam,
        jump_mu,
        jump_sigma,
    )

    kappa = np.exp(
        jump_mu + 0.5 * jump_sigma**2
    ) - 1

    k = np.arange(40)

    means = (
        (mu - lam * kappa - 0.5 * sigma**2)
        * dt
        + k * jump_mu
    )

    variances = (
        sigma**2 * dt
        + k * jump_sigma**2
    )

    log_weights = poisson.logpmf(k, lam * dt)

    expected = 0.0

    for r in returns:
        terms = (
            log_weights
            + norm.logpdf(
                r,
                loc=means,
                scale=np.sqrt(variances),
            )
        )

        expected += logsumexp(terms)

    assert calculated == pytest.approx(
        expected, abs=1e-9
    )

@pytest.mark.parametrize(
    "sigma,lam,jump_sigma",
    [
        (0.0, 2.0, 0.1),
        (-0.2, 2.0, 0.1),
        (0.2, -1.0, 0.1),
        (0.2, 2.0, -0.1),
    ],
)
def test_invalid_merton_parameters(
    sigma, lam, jump_sigma
):
    with pytest.raises(ValueError):
        MertonCalibrator.log_likelihood(
            [0.01, -0.02, 0.03],
            dt=1 / 252,
            mu=0.08,
            sigma=sigma,
            jump_lambda=lam,
            jump_mu=-0.05,
            jump_sigma=jump_sigma,
        )

def test_finite_merton_likelihood():
    rng = np.random.default_rng(42)

    returns = rng.normal(
        0.0, 0.02, size=1000
    )

    value = MertonCalibrator.log_likelihood(
        returns,
        dt=1 / 252,
        mu=0.08,
        sigma=0.2,
        jump_lambda=2.0,
        jump_mu=-0.05,
        jump_sigma=0.1,
    )

    assert np.isfinite(value)

def test_fitted_likelihood():
    model = make_merton()

    dataset = SyntheticDataGenerator.generate(
        model,
        SimulationConfig(
            horizon=1.0,
            steps=12,
            paths=300,
            seed=42,
        ),
    )

    result = MertonCalibrator.fit(
        dataset.log_returns,
        dt=dataset.dt,
        n_starts=3,
        seed=42,
        maxiter=400,
    )

    assert result.converged
    assert np.isfinite(result.log_likelihood)

    poor_ll = MertonCalibrator.log_likelihood(
        dataset.log_returns,
        dt=dataset.dt,
        mu=0.0,
        sigma=0.5,
        jump_lambda=0.0,
        jump_mu=0.0,
        jump_sigma=0.0,
    )

    assert result.log_likelihood > poor_ll

def test_merton_fit_constraints():
    model = make_merton()

    dataset = SyntheticDataGenerator.generate(
        model,
        SimulationConfig(
            horizon=1.0,
            steps=12,
            paths=200,
            seed=7,
        ),
    )

    result = MertonCalibrator.fit(
        dataset.log_returns,
        dt=dataset.dt,
        n_starts=2,
        seed=7,
        maxiter=300,
    )

    params = result.parameters

    assert params["sigma"] > 0
    assert params["jump_lambda"] >= 0
    assert params["jump_sigma"] >= 0
    assert all(np.isfinite(v) for v in params.values())

def test_merton_calibration_reproducibility():
    model = make_merton()

    dataset = SyntheticDataGenerator.generate(
        model,
        SimulationConfig(
            horizon=1.0,
            steps=12,
            paths=100,
            seed=123,
        ),
    )

    kwargs = dict(
        log_returns=dataset.log_returns,
        dt=dataset.dt,
        n_starts=2,
        seed=99,
        maxiter=300,
    )

    first = MertonCalibrator.fit(**kwargs)
    second = MertonCalibrator.fit(**kwargs)

    for key in first.parameters:
        assert first.parameters[key] == pytest.approx(
            second.parameters[key],
            abs=1e-10,
        )

def test_merton_result_metadata():
    model = make_merton()

    dataset = SyntheticDataGenerator.generate(
        model,
        SimulationConfig(
            horizon=1.0,
            steps=12,
            paths=100,
            seed=42,
        ),
    )

    result = MertonCalibrator.fit(
        dataset.log_returns,
        dt=dataset.dt,
        n_starts=2,
        maxiter=300,
    )

    assert result.model_name == "MertonAssetPriceModel"
    assert result.n_parameters == 5
    assert result.n_observations == 1200
    assert result.converged
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)

def test_invalid_optimization_settings():
    returns = np.array([
        0.01, -0.02, 0.03
    ])

    with pytest.raises(ValueError):
        MertonCalibrator.fit(
            returns,
            dt=1 / 252,
            n_starts=0,
        )

    with pytest.raises(ValueError):
        MertonCalibrator.fit(
            returns,
            dt=1 / 252,
            maxiter=0,
        )


def test_fixed_cutoff_controls_poisson_tail():
    dt = 0.25
    max_lambda = 20.0
    tolerance = 1e-12

    components = MertonCalibrator._component_count(
        jump_lambda=max_lambda,
        dt=dt,
        tail_tolerance=tolerance,
        max_components=1000,
    )

    for lam in (0.0, 1.0, 5.0, 10.0, 20.0):
        omitted_mass = poisson.sf(
            components - 1,
            lam * dt,
        )

        assert omitted_mass <= tolerance


def test_fixed_and_adaptive_likelihood_agree():
    returns = np.array([
        -0.15, -0.05, 0.01, 0.03, 0.12
    ])

    dt = 0.25

    parameters = dict(
        mu=0.08,
        sigma=0.2,
        jump_lambda=3.0,
        jump_mu=-0.1,
        jump_sigma=0.15,
    )

    fixed_components = MertonCalibrator._component_count(
        jump_lambda=20.0,
        dt=dt,
        tail_tolerance=1e-12,
        max_components=1000,
    )

    adaptive = MertonCalibrator.log_likelihood(
        returns,
        dt,
        **parameters,
    )

    fixed = MertonCalibrator.log_likelihood(
        returns,
        dt,
        **parameters,
        _fixed_components=fixed_components,
    )

    assert fixed == pytest.approx(
        adaptive,
        abs=1e-9,
    )


@pytest.mark.parametrize(
    "tolerance",
    [0.0, 1.0, -1e-5, np.nan, np.inf],
)
def test_invalid_tail_tolerance(tolerance):
    returns = np.array([0.01, -0.02, 0.03])

    with pytest.raises(ValueError):
        MertonCalibrator.log_likelihood(
            returns,
            dt=1 / 252,
            mu=0.08,
            sigma=0.2,
            jump_lambda=2.0,
            jump_mu=-0.1,
            jump_sigma=0.1,
            tail_tolerance=tolerance,
        )

    with pytest.raises(ValueError):
        MertonCalibrator.fit(
            returns,
            dt=1 / 252,
            tail_tolerance=tolerance,
        )


def test_fixed_cutoff_rejects_invalid_values():
    returns = np.array([0.01, -0.02, 0.03])

    for invalid in (0, -1, 1.5, True, 1001):
        with pytest.raises(ValueError):
            MertonCalibrator.log_likelihood(
                returns,
                dt=1 / 252,
                mu=0.08,
                sigma=0.2,
                jump_lambda=2.0,
                jump_mu=-0.1,
                jump_sigma=0.1,
                _fixed_components=invalid,
            )


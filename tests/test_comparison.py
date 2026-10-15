import numpy as np
import pytest

from stochastic_inference_engine import (
    GBMCalibrator,
    GeometricBrownianMotion,
    ModelComparator,
    ModelComparisonResult,
    ModelScore,
    SimulationConfig,
    SyntheticDataGenerator,
)


def make_returns(seed=42):
    model = GeometricBrownianMotion(
        x0=100.0,
        mu=0.08,
        sigma=0.20,
    )

    dataset = SyntheticDataGenerator.generate(
        model,
        SimulationConfig(
            horizon=1.0,
            steps=300,
            paths=1,
            seed=seed,
        ),
    )

    return dataset.log_returns[:, 0], dataset.dt

def test_comparison_split():
    returns, dt = make_returns()

    result = ModelComparator.compare(
        returns,
        dt,
        train_fraction=0.8,
        models=("gbm",),
    )

    assert result.n_train == 240
    assert result.n_test == 60
    assert result.dt == pytest.approx(dt)

def test_gbm_test_likelihood():
    returns, dt = make_returns()

    result = ModelComparator.compare(
        returns,
        dt,
        train_fraction=0.8,
        models=("gbm",),
    )

    score = result.scores[0]

    direct_fit = GBMCalibrator.fit(
        returns[:240], dt
    )

    p = direct_fit.parameters

    expected = GBMCalibrator.log_likelihood(
        returns[240:],
        dt=dt,
        mu=p["mu"],
        sigma=p["sigma"],
    )

    assert score.test_log_likelihood == pytest.approx(
        expected
    )

    assert score.test_average_log_likelihood == (
        pytest.approx(expected / 60)
    )

def test_ranking_uses_test_score():
    returns, dt = make_returns()

    fitted = GBMCalibrator.fit(
        returns[:240], dt
    )

    low = ModelScore(
        model_name="ModelA",
        calibration=fitted,
        test_log_likelihood=10.0,
        test_average_log_likelihood=0.1,
    )

    high = ModelScore(
        model_name="ModelB",
        calibration=fitted,
        test_log_likelihood=20.0,
        test_average_log_likelihood=0.2,
    )

    comparison = ModelComparisonResult(
        scores=(low, high),
        n_train=240,
        n_test=60,
        dt=dt,
    )

    assert comparison.best_model == "ModelB"
    assert comparison.ranking()[0].model_name == "ModelB"

def test_comparison_reproducibility():
    returns, dt = make_returns()

    first = ModelComparator.compare(
        returns, dt, models=("gbm",)
    )

    second = ModelComparator.compare(
        returns, dt, models=("gbm",)
    )

    assert (
        first.scores[0].test_average_log_likelihood
        == second.scores[0].test_average_log_likelihood
    )

def test_comparison_rejects_2d_input():
    returns, dt = make_returns()

    with pytest.raises(ValueError):
        ModelComparator.compare(
            returns[:, None], dt
        )

def test_invalid_model_names():
    returns, dt = make_returns()

    with pytest.raises(ValueError):
        ModelComparator.compare(
            returns, dt, models=("unknown",)
        )

    with pytest.raises(ValueError):
        ModelComparator.compare(
            returns, dt, models=("gbm", "gbm")
        )

@pytest.mark.parametrize(
    "fraction",
    [0.0, 1.0, -0.5, 1.5, np.nan],
)
def test_invalid_train_fraction(fraction):
    returns, dt = make_returns()

    with pytest.raises(ValueError):
        ModelComparator.compare(
            returns,
            dt,
            train_fraction=fraction,
        )

def test_comparison_information_criteria():
    returns, dt = make_returns()

    result = ModelComparator.compare(
        returns,
        dt,
        models=("gbm",),
    )

    score = result.scores[0]

    assert score.train_aic == pytest.approx(
        score.calibration.aic
    )

    assert score.train_bic == pytest.approx(
        score.calibration.bic
    )

def test_both_models_comparison():
    # Heavy-tailed observations give the jump model
    rng = np.random.default_rng(123)

    returns = rng.normal(0.0003, 0.01, size=160)

    jump_indices = np.array(
        [12, 28, 45, 67, 92, 117, 142]
    )

    returns[jump_indices] -= 0.08

    result = ModelComparator.compare(
        returns,
        dt=1 / 252,
        models=("gbm", "merton"),
        merton_n_starts=2,
        merton_seed=42,
        merton_maxiter=300,
    )

    assert len(result.scores) == 2

    assert {
        score.model_name for score in result.scores
    } == {
        "GeometricBrownianMotion",
        "MertonAssetPriceModel",
    }

    assert all(
        np.isfinite(score.test_average_log_likelihood)
        for score in result.scores
    )

    assert result.best_model in {
        "GeometricBrownianMotion",
        "MertonAssetPriceModel",
    }

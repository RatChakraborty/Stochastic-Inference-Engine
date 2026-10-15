import numpy as np
import pytest

from stochastic_inference_engine import (
    StatisticalEvaluator,
    RiskEvaluator,
    DistributionEvaluator,
)

def test_return_statistics():
    values = np.array([-2., -1., 0., 1., 2.])
    result = StatisticalEvaluator.summarize(values)

    assert result.n_observations == 5
    assert result.mean == pytest.approx(0.0)
    assert result.variance == pytest.approx(2.5)
    assert result.skewness == pytest.approx(0.0)


def test_normal_return_statistics():
    rng = np.random.default_rng(42)
    values = rng.normal(size=50_000)
    result = StatisticalEvaluator.summarize(values)

    assert abs(result.skewness) < 0.05
    assert abs(result.excess_kurtosis) < 0.10


def test_autocorrelation():
    rng = np.random.default_rng(42)
    values = rng.normal(size=20_000)

    assert abs(
        StatisticalEvaluator.autocorrelation(values, lag=1)
    ) < 0.04


def test_autocorrelation_does_not_cross_paths():
    values = np.array([
        [1., 10.],
        [2., 11.],
        [3., 12.],
    ])
    result = StatisticalEvaluator.autocorrelation(
        values, lag=1
    )
    assert np.isfinite(result)


def test_historical_var():
    returns = np.array([-0.10, -0.05, 0.01, 0.02])
    expected = np.quantile(-returns, 0.75)

    assert RiskEvaluator.value_at_risk(
        returns, confidence=0.75
    ) == pytest.approx(expected)


def test_expected_shortfall():
    returns = np.array([-0.10, -0.05, 0.01, 0.02])

    # Worst 25% of four observations is one loss.
    assert RiskEvaluator.expected_shortfall(
        returns, confidence=0.75
    ) == pytest.approx(0.10)


def test_maximum_drawdown():
    prices = np.array([100., 120., 90., 110.])

    result = RiskEvaluator.maximum_drawdown(prices)

    assert result.shape == (1,)
    assert result[0] == pytest.approx(0.25)


def test_distribution_identical_samples():
    values = np.array([-0.03, 0.01, 0.02, 0.04])

    result = DistributionEvaluator.compare(
        values, values
    )

    assert result.ks_statistic == pytest.approx(0)
    assert result.wasserstein_distance == pytest.approx(0)


def test_distribution_shift():
    reference = np.array([0., 1., 2., 3.])
    shifted = reference + 2

    result = DistributionEvaluator.compare(
        reference, shifted
    )

    assert result.wasserstein_distance == pytest.approx(2.0)
    assert result.ks_statistic > 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        StatisticalEvaluator.summarize([np.nan, 1.0])

    with pytest.raises(ValueError):
        RiskEvaluator.maximum_drawdown([100., -10.])

    with pytest.raises(ValueError):
        RiskEvaluator.value_at_risk(
            [0.01, -0.02], confidence=1.0
        )


def test_expected_shortfall_fractional_tail():
    returns = np.array([-0.10, -0.05, 0.01, 0.02])

    # Worst 37.5% = worst 1.5 observations.
    expected = (0.10 + 0.5 * 0.05) / 1.5

    assert RiskEvaluator.expected_shortfall(
        returns, confidence=0.625
    ) == pytest.approx(expected)

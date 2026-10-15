
import numpy as np
import pytest

from stochastic_inference_engine.comparison.comparator import (
    ModelComparator,
)


def test_merton_failure_preserves_gbm(monkeypatch):
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, size=100)

    def failing_fit(*args, **kwargs):
        raise RuntimeError("Simulated Merton failure")

    monkeypatch.setattr(
        "stochastic_inference_engine.comparison.comparator."
        "MertonCalibrator.fit",
        failing_fit,
    )

    result = ModelComparator.compare(
        returns,
        dt=1 / 252,
    )

    assert len(result.scores) == 1
    assert result.best_model == "GeometricBrownianMotion"

    assert len(result.failures) == 1
    assert result.failures[0].model_name == "merton"
    assert result.failures[0].stage == "calibration"
    assert "Simulated Merton failure" in (
        result.failures[0].message
    )


def test_both_models_fail(monkeypatch):
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, size=100)

    def failing_fit(*args, **kwargs):
        raise RuntimeError("Simulated calibration failure")

    monkeypatch.setattr(
        "stochastic_inference_engine.comparison.comparator."
        "GBMCalibrator.fit",
        failing_fit,
    )

    monkeypatch.setattr(
        "stochastic_inference_engine.comparison.comparator."
        "MertonCalibrator.fit",
        failing_fit,
    )

    with pytest.raises(
        RuntimeError,
        match="All requested model comparisons failed",
    ):
        ModelComparator.compare(
            returns,
            dt=1 / 252,
        )


def test_nonfinite_test_likelihood_is_failure(monkeypatch):
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, size=100)

    def nonfinite_likelihood(*args, **kwargs):
        return float("nan")

    monkeypatch.setattr(
        "stochastic_inference_engine.comparison.comparator."
        "MertonCalibrator.log_likelihood",
        nonfinite_likelihood,
    )

    result = ModelComparator.compare(
        returns,
        dt=1 / 252,
    )

    assert len(result.scores) == 1
    assert result.best_model == "GeometricBrownianMotion"

    assert len(result.failures) == 1
    assert result.failures[0].model_name == "merton"
    assert result.failures[0].stage == "evaluation"

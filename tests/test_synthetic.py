import numpy as np
import pytest

from stochastic_inference_engine import (
    GeometricBrownianMotion,
    SimulationConfig,
    SyntheticDataGenerator,
    SyntheticDataset,
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

def test_model_parameters_are_immutable():
    model = make_gbm()
    config = make_config(paths=10, steps=10)

    dataset = SyntheticDataGenerator.generate(
        model, config
    )

    original_mu = dataset.model_parameters["mu"]

    with pytest.raises(TypeError):
        dataset.model_parameters["mu"] = 999.0

    assert dataset.model_parameters["mu"] == original_mu


def test_model_parameters_are_isolated_from_caller():
    model = make_gbm()
    config = make_config(paths=10, steps=10)

    original = SyntheticDataGenerator.generate(
        model, config
    )

    parameters = dict(original.model_parameters)

    dataset = SyntheticDataset(
        simulation=original.simulation,
        model_parameters=parameters,
    )

    original_mu = parameters["mu"]
    parameters["mu"] = 999.0

    assert dataset.model_parameters["mu"] == original_mu


def test_small_return_numerical_precision():
    prices = np.array([
        [100.0],
        [100.0 + 1e-10],
    ])

    expected = np.log(prices[1, 0]) - np.log(prices[0, 0])

    actual = np.log1p(
        (prices[1, 0] - prices[0, 0]) / prices[0, 0]
    )

    assert np.isclose(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-12,
    )



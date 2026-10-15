import numpy as np
import pytest

from stochastic_inference_engine import (
    SimulationConfig,
    SimulationResult,
    SimulationEngine,
    EulerMaruyamaModel,
    EulerMaruyamaSimulator,
    GeometricBrownianMotion,
    MertonAssetPriceModel,
)


def test_config_time_grid():
    config = SimulationConfig(
        horizon=1.0,
        steps=4,
        paths=10,
        seed=42,
    )

    assert config.dt == pytest.approx(0.25)

    np.testing.assert_allclose(
        config.time_grid,
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )

def test_config_invalid_values():
    with pytest.raises(ValueError):
        SimulationConfig(0.0, 10, 10)

    with pytest.raises(ValueError):
        SimulationConfig(1.0, 0, 10)

    with pytest.raises(ValueError):
        SimulationConfig(1.0, 10, 0)

    with pytest.raises(ValueError):
        SimulationConfig(1.0, 10, 10, seed=-1)

def test_result_statistics():
    values = np.array([
        [100.0, 100.0],
        [110.0, 90.0],
        [120.0, 80.0],
    ])

    result = SimulationResult(
        time_grid=np.array([0.0, 0.5, 1.0]),
        values=values,
        model_name="TestModel",
    )

    np.testing.assert_allclose(
        result.mean_path(),
        [100.0, 100.0, 100.0],
    )

    np.testing.assert_allclose(
        result.variance_path(),
        [0.0, 200.0, 800.0],
    )

    np.testing.assert_allclose(
        result.terminal_values,
        [120.0, 80.0],
    )

    assert result.terminal_mean() == pytest.approx(100.0)
    assert result.n_paths == 2
    assert result.n_steps == 2

def test_result_input_copy():
    original = np.array([
        [1.0, 1.0],
        [2.0, 3.0],
    ])

    result = SimulationResult(
        time_grid=np.array([0.0, 1.0]),
        values=original,
        model_name="TestModel",
    )

    original[1, 0] = 999.0

    assert result.values[1, 0] == 2.0

def test_gbm_engine_reproducibility():
    model = GeometricBrownianMotion(
        x0=100.0,
        mu=0.05,
        sigma=0.2,
    )

    config = SimulationConfig(
        horizon=1.0,
        steps=64,
        paths=100,
        seed=42,
    )

    first = SimulationEngine.run(model, config)
    second = SimulationEngine.run(model, config)

    np.testing.assert_array_equal(
        first.values, second.values
    )

    assert first.model_name == "GeometricBrownianMotion"
    assert first.values.shape == (65, 100)

def test_gbm_engine_expected_value():
    model = GeometricBrownianMotion(
        x0=100.0,
        mu=0.05,
        sigma=0.2,
    )

    config = SimulationConfig(
        horizon=1.0,
        steps=1,
        paths=50_000,
        seed=42,
    )

    result = SimulationEngine.run(model, config)

    expected = model.x0 * np.exp(
        model.mu * config.horizon
    )

    assert abs(
        result.terminal_mean() - expected
    ) < 5 * result.terminal_standard_error()

def test_euler_adapter():
    solver = EulerMaruyamaSimulator(
        drift_func=lambda x, t: 0.05 * x,
        diff_func=lambda x, t: 0.2 * x,
    )

    model = EulerMaruyamaModel(
        simulator=solver,
        x0=100.0,
    )

    config = SimulationConfig(
        horizon=1.0,
        steps=100,
        paths=50,
        seed=42,
    )

    result = SimulationEngine.run(model, config)

    assert result.values.shape == (101, 50)
    assert np.all(result.values[0] == 100.0)

def test_merton_asset_model_engine():
    model = MertonAssetPriceModel(
        x0=100.0,
        mu=0.05,
        sigma=0.2,
        jump_lambda=2.0,
        jump_mu=-0.05,
        jump_sigma=0.1,
    )

    config = SimulationConfig(
        horizon=1.0,
        steps=100,
        paths=100,
        seed=42,
    )

    result = SimulationEngine.run(model, config)

    assert result.values.shape == (101, 100)
    assert np.all(result.values > 0)

def test_engine_rejects_invalid_output():
    class BrokenModel:
        def simulate(self, horizon, steps, paths, seed=None):
            return np.zeros((steps, paths))

    config = SimulationConfig(1.0, 10, 5)

    with pytest.raises(ValueError):
        SimulationEngine.run(BrokenModel(), config)

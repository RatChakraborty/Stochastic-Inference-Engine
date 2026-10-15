import numpy as np
import pytest

from stochastic_inference_engine.dynamics.sdes import (
    EulerMaruyamaSimulator,
)


def make_gbm_solver(mu=0.05, sigma=0.2):
    return EulerMaruyamaSimulator(
        drift_func=lambda x, t: mu * x,
        diff_func=lambda x, t: sigma * x,
    )


def test_shape_and_initial_value():
    solver = make_gbm_solver()

    result = solver.simulate(
        x0=100.0,
        horizon=1.0,
        steps=100,
        paths=20,
        seed=42,
    )

    assert result.shape == (101, 20)
    assert np.all(result[0] == 100.0)


def test_reproducibility():
    solver = make_gbm_solver()

    first = solver.simulate(
        100.0, 1.0, 100, 20, seed=42
    )

    second = solver.simulate(
        100.0, 1.0, 100, 20, seed=42
    )

    np.testing.assert_array_equal(first, second)


def test_deterministic_drift():
    solver = EulerMaruyamaSimulator(
        drift_func=lambda x, t: 2.0,
        diff_func=lambda x, t: 0.0,
    )

    result = solver.simulate(
        x0=3.0,
        horizon=2.0,
        steps=100,
        paths=10,
        seed=42,
    )

    np.testing.assert_allclose(
        result[-1],
        7.0,
        rtol=0,
        atol=1e-12,
    )


def test_external_brownian_increments():
    solver = EulerMaruyamaSimulator(
        drift_func=lambda x, t: 0.0,
        diff_func=lambda x, t: 1.0,
    )

    increments = np.array([
        [0.1, -0.2],
        [0.3, 0.4],
        [-0.1, 0.2],
    ])

    result = solver.simulate(
        x0=0.0,
        horizon=1.0,
        steps=3,
        paths=2,
        brownian_increments=increments,
    )

    expected = np.vstack([
        np.zeros(2),
        np.cumsum(increments, axis=0),
    ])

    np.testing.assert_allclose(result, expected)


def test_invalid_inputs():
    solver = make_gbm_solver()

    with pytest.raises(ValueError):
        solver.simulate(
            100.0, 1.0, 0, 10, seed=42
        )

    with pytest.raises(ValueError):
        solver.simulate(
            100.0, -1.0, 100, 10, seed=42
        )

    with pytest.raises(ValueError):
        solver.simulate(
            100.0, 1.0, 100, 10,
            brownian_increments=np.zeros((10, 10)),
        )

    with pytest.raises(ValueError):
        solver.simulate(
            100.0, 1.0, 100, 10,
            seed=42,
            brownian_increments=np.zeros((100, 10)),
        )


def test_strong_convergence():
    x0 = 100.0
    mu = 0.05
    sigma = 0.3
    T = 1.0

    n_paths = 5000
    finest_steps = 256

    rng = np.random.default_rng(42)

    fine_dw = rng.standard_normal(
        (finest_steps, n_paths)
    ) * np.sqrt(T / finest_steps)

    # Exact terminal values using the same Brownian paths
    w_terminal = fine_dw.sum(axis=0)

    exact_terminal = x0 * np.exp(
        (mu - 0.5 * sigma**2) * T
        + sigma * w_terminal
    )

    solver = make_gbm_solver(mu, sigma)

    step_counts = [8, 16, 32, 64, 128, 256]
    errors = []

    for steps in step_counts:
        block_size = finest_steps // steps

        # Combine fine increments into coarse increments
        coarse_dw = fine_dw.reshape(
            steps, block_size, n_paths
        ).sum(axis=1)

        approximate = solver.simulate(
            x0=x0,
            horizon=T,
            steps=steps,
            paths=n_paths,
            brownian_increments=coarse_dw,
        )

        error = np.mean(
            np.abs(approximate[-1] - exact_terminal)
        )

        errors.append(error)

    # The finest approximation should be more accurate
    assert errors[-1] < errors[0]

    # Estimate empirical convergence order
    dt_values = T / np.array(step_counts)

    slope = np.polyfit(
        np.log(dt_values),
        np.log(errors),
        deg=1,
    )[0]

    # Conservative range for finite-sample convergence
    assert 0.25 < slope < 0.85


def test_weak_convergence_order():
    x0 = 100.0
    mu = 0.4
    T = 1.0

    exact_mean = x0 * np.exp(mu * T)

    step_counts = np.array([4, 8, 16, 32, 64, 128])

    errors = []

    for steps in step_counts:
        em_mean = x0 * (1 + mu * T / steps)**steps

        errors.append(abs(em_mean - exact_mean))

    dt_values = T / step_counts

    slope = np.polyfit(
        np.log(dt_values),
        np.log(errors),
        deg=1,
    )[0]

    # Expected first-order weak convergence
    assert 0.9 < slope < 1.1


def test_monte_carlo_terminal_mean():
    x0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    steps = 64
    n_paths = 30_000

    solver = make_gbm_solver(mu, sigma)

    result = solver.simulate(
        x0=x0,
        horizon=T,
        steps=steps,
        paths=n_paths,
        seed=123,
    )

    terminal = result[-1]

    # Exact expectation of the Euler-Maruyama scheme
    em_expected_mean = x0 * (
        1 + mu * T / steps
    )**steps

    standard_error = (
        terminal.std(ddof=1) / np.sqrt(n_paths)
    )

    assert abs(
        terminal.mean() - em_expected_mean
    ) < 5 * standard_error


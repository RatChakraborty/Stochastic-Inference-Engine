import numpy as np
import pytest
from stochastic_inference_engine import (
    FFTDensityRecovery,
    GeometricBrownianMotion,
    Ising1DSampler,
    MonteCarloIntegrator,
)


def test_gbm_simulation_shape():
    gbm = GeometricBrownianMotion(x0=100.0, mu=0.05, sigma=0.2)
    paths = gbm.simulate(horizon=1.0, steps=100, paths=10)
    assert paths.shape == (101, 10)
    assert np.all(paths[0] == 100.0)


def test_fft_density_recovery_mse():
    _, _, _, mse = FFTDensityRecovery.recover_gbm(
        x0=100.0, T=1.0, mu=0.05, sigma=0.2, grid_size=2048
    )
    assert mse < 1e-4


def test_ising_sampler_output():
    ising = Ising1DSampler(num_spins=20)
    spins, mag = ising.run_metropolis(steps=200, beta=0.4)
    assert spins.shape == (201, 20)
    assert len(mag) == 201


def test_monte_carlo_integration():
    func = lambda x: x**2
    est, se = MonteCarloIntegrator.integrate_1d(
        func, a=0.0, b=1.0, n_samples=100_000
    )
    assert abs(est - (1.0 / 3.0)) < 3 * se
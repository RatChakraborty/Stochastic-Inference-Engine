import numpy as np
from stochastic_inference_engine import (
    FFTDensityRecovery,
    GeometricBrownianMotion,
    Ising1DSampler,
    MonteCarloIntegrator,
)


def main():
    print("1. Simulating Geometric Brownian Motion")
    gbm = GeometricBrownianMotion(x0=100.0, mu=0.05, sigma=0.2)
    paths = gbm.simulate(horizon=1.0, steps=252, paths=5)
    print(f"Generated {paths.shape[1]} paths over {paths.shape[0]} steps.")
    print(f"Final Path Values: {np.round(paths[-1], 2)}\n")

    print("2. Performing FFT Density Recovery")
    values, recovered_pdf, true_pdf, mse = FFTDensityRecovery.recover_gbm(
        x0=100.0, T=1.0, mu=0.05, sigma=0.2, grid_size=2048
    )
    print(f"FFT Inversion Completed, Mean Squared Error (MSE): {mse:.2e}\n")

    print("3. Running 1D Ising Metropolis Sampler")
    ising = Ising1DSampler(num_spins=30)
    spins, mag = ising.run_metropolis(steps=1000, beta=0.5)
    print(
        f"MCMC Chain Completed, Final Mean Magnetization: {mag[-1]:.4f}\n"
    )

    print("4. Monte Carlo Integration")

    def target_func(x):
        return x**2

    est, se = MonteCarloIntegrator.integrate_1d(
        target_func, a=0.0, b=2.0, n_samples=50_000
    )
    print(f"Integral of x^2 over [0, 2]: {est:.5f} (Std Error: {se:.5f})")
    print(f"Exact Analytical Result : {8.0 / 3.0:.5f}")


if __name__ == "__main__":
    main()
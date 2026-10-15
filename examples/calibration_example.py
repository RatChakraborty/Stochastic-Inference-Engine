
from stochastic_inference_engine import (
    GeometricBrownianMotion,
    SimulationConfig,
    SyntheticDataGenerator,
    GBMCalibrator,
)


def main():
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

    print("\nGBM CALIBRATION RESULTS")
    print("=" * 45)

    print(f"Observations: {result.n_observations}")
    print(f"Method: {result.method}")
    print(f"Converged: {result.converged}")

    print("\nTrue vs. estimated parameters")
    print("-" * 45)

    print(
        f"Drift:      true={true_mu:.6f}, "
        f"estimated={result.parameters['mu']:.6f}"
    )

    print(
        f"Volatility: true={true_sigma:.6f}, "
        f"estimated={result.parameters['sigma']:.6f}"
    )

    print("\nApproximate standard errors")
    print("-" * 45)

    print(
        f"Drift SE:      "
        f"{result.standard_errors['mu']:.6f}"
    )

    print(
        f"Volatility SE: "
        f"{result.standard_errors['sigma']:.6f}"
    )

    print("\nModel fit")
    print("-" * 45)

    print(f"Log-likelihood: {result.log_likelihood:.4f}")
    print(f"AIC: {result.aic:.4f}")
    print(f"BIC: {result.bic:.4f}")


if __name__ == "__main__":
    main()

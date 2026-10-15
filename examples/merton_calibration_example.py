from stochastic_inference_engine import (
    MertonAssetPriceModel,
    MertonCalibrator,
    SimulationConfig,
    SyntheticDataGenerator,
)


def main():
    model = MertonAssetPriceModel(
        x0=100.0,
        mu=0.08,
        sigma=0.20,
        jump_lambda=4.0,
        jump_mu=-0.12,
        jump_sigma=0.12,
    )

    config = SimulationConfig(
        horizon=1.0,
        steps=12,
        paths=500,
        seed=42,
    )

    dataset = SyntheticDataGenerator.generate(
        model, config
    )

    result = MertonCalibrator.fit(
        dataset.log_returns,
        dt=dataset.dt,
        n_starts=5,
        seed=42,
        maxiter=500,
    )

    true_parameters = {
        "mu": 0.08,
        "sigma": 0.20,
        "jump_lambda": 4.0,
        "jump_mu": -0.12,
        "jump_sigma": 0.12,
    }

    print("\nMERTON CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Converged: {result.converged}")
    print(f"Observations: {result.n_observations}")
    print(f"Method: {result.method}")
    print(result.message)

    print("\nParameter             True        Estimated")
    print("-" * 60)

    for name, true_value in true_parameters.items():
        estimated = result.parameters[name]

        print(
            f"{name:<20}"
            f"{true_value:>10.5f}"
            f"{estimated:>15.5f}"
        )

    print("\nMODEL FIT")
    print("-" * 60)

    print(
        f"Log-likelihood: {result.log_likelihood:.4f}"
    )
    print(f"AIC: {result.aic:.4f}")
    print(f"BIC: {result.bic:.4f}")


if __name__ == "__main__":
    main()

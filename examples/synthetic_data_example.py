import numpy as np

from stochastic_inference_engine import (
    GeometricBrownianMotion,
    MertonAssetPriceModel,
    SimulationConfig,
    SyntheticDataGenerator,
)

def main():
    config = SimulationConfig(
        horizon=1.0,
        steps=252,
        paths=1000,
        seed=42,
    )

    models = [
        GeometricBrownianMotion(
            x0=100.0,
            mu=0.08,
            sigma=0.2,
        ),
        MertonAssetPriceModel(
            x0=100.0,
            mu=0.08,
            sigma=0.2,
            jump_lambda=2.0,
            jump_mu=-0.05,
            jump_sigma=0.1,
        ),
    ]

    for model in models:
        dataset = SyntheticDataGenerator.generate(
            model, config
        )

        print(f"\nModel: {dataset.model_name}")
        print(f"Price shape: {dataset.prices.shape}")
        print(
            f"Log-return shape: "
            f"{dataset.log_returns.shape}"
        )

        print(
            "Mean terminal price:",
            dataset.terminal_summary()["mean"],
        )

        print(
            "Annualized log-return volatility:",
            dataset.annualized_volatility(252),
        )

        print(
            "First five prices from path 0:",
            dataset.prices[:5, 0],
        )

        print(
            "First five log returns from path 0:",
            dataset.log_returns[:5, 0],
        )

if __name__ == "__main__":
    main()


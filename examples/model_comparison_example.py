from stochastic_inference_engine import (
    MertonAssetPriceModel,
    SimulationConfig,
    SyntheticDataGenerator,
    StatisticalEvaluator,
    RiskEvaluator,
    ModelComparator,
)


def main():
    true_model = MertonAssetPriceModel(
        x0=100.0,
        mu=0.08,
        sigma=0.20,
        jump_lambda=4.0,
        jump_mu=-0.12,
        jump_sigma=0.12,
    )

    config = SimulationConfig(
        horizon=10.0,
        steps=2520,
        paths=1,
        seed=42,
    )

    dataset = SyntheticDataGenerator.generate(
        true_model, config
    )

    returns = dataset.log_returns[:, 0]
    prices = dataset.prices[:, 0]

    print("\nREFERENCE DATA STATISTICS")
    print("=" * 55)

    statistics = StatisticalEvaluator.summarize(returns)

    print(f"Observations: {statistics.n_observations}")
    print(f"Mean return: {statistics.mean:.6f}")
    print(f"Volatility per step: {statistics.std:.6f}")
    print(f"Skewness: {statistics.skewness:.4f}")
    print(
        f"Excess kurtosis: "
        f"{statistics.excess_kurtosis:.4f}"
    )

    print(
        "95% historical log-loss VaR:",
        f"{RiskEvaluator.value_at_risk(returns):.6f}",
    )

    print(
        "95% historical log-loss ES:",
        f"{RiskEvaluator.expected_shortfall(returns):.6f}",
    )

    print(
        "Maximum drawdown:",
        f"{RiskEvaluator.maximum_drawdown(prices)[0]:.4%}",
    )

    print("\nCALIBRATING AND COMPARING MODELS")
    print("=" * 55)

    result = ModelComparator.compare(
        log_returns=returns,
        dt=dataset.dt,
        train_fraction=0.8,
        models=("gbm", "merton"),
        merton_n_starts=5,
        merton_seed=42,
        merton_maxiter=500,
    )

    print(f"Training observations: {result.n_train}")
    print(f"Test observations: {result.n_test}")

    for rank, score in enumerate(result.ranking(), 1):
        print(f"\nRank {rank}: {score.model_name}")
        print("-" * 55)

        print(
            "Test average log-likelihood:",
            f"{score.test_average_log_likelihood:.6f}",
        )

        print(
            "Test total log-likelihood:",
            f"{score.test_log_likelihood:.4f}",
        )

        print(f"Training AIC: {score.train_aic:.4f}")
        print(f"Training BIC: {score.train_bic:.4f}")

        print("Fitted parameters:")

        for name, value in score.calibration.parameters.items():
            print(f"  {name}: {value:.6f}")

    print("\nBEST MODEL BY HELD-OUT LOG-LIKELIHOOD")
    print("=" * 55)
    print(result.best_model)

if __name__ == "__main__":
    main()

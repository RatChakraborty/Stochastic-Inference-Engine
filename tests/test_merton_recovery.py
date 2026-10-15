import numpy as np
import pytest

from stochastic_inference_engine import (
    MertonCalibrator,
    MertonAssetPriceModel,
    SimulationConfig,
    SyntheticDataGenerator,
)


PARAMETER_NAMES = (
    "mu",
    "sigma",
    "jump_lambda",
    "jump_mu",
    "jump_sigma",
)


REGIMES = {
    "low": {
        "mu": 0.08,
        "sigma": 0.20,
        "jump_lambda": 0.5,
        "jump_mu": -0.12,
        "jump_sigma": 0.12,
    },
    "moderate": {
        "mu": 0.08,
        "sigma": 0.20,
        "jump_lambda": 4.0,
        "jump_mu": -0.12,
        "jump_sigma": 0.12,
    },
    "high": {
        "mu": 0.08,
        "sigma": 0.20,
        "jump_lambda": 12.0,
        "jump_mu": -0.12,
        "jump_sigma": 0.12,
    },
}


def run_recovery_experiment(
    regime,
    n_paths,
    seed,
):
    true_parameters = REGIMES[regime]

    model = MertonAssetPriceModel(
        x0=100.0,
        **true_parameters,
    )

    dataset = SyntheticDataGenerator.generate(
        model,
        SimulationConfig(
            horizon=1.0,
            steps=12,
            paths=n_paths,
            seed=seed,
        ),
    )

    result = MertonCalibrator.fit(
        dataset.log_returns,
        dt=dataset.dt,
        n_starts=3,
        seed=seed,
        maxiter=400,
    )

    estimates = result.parameters

    errors = {
        name: (
            estimates[name] - true_parameters[name]
        )
        for name in PARAMETER_NAMES
    }

    return {
        "regime": regime,
        "n_paths": n_paths,
        "seed": seed,
        "n_observations": result.n_observations,
        "converged": result.converged,
        "true": true_parameters,
        "estimated": estimates,
        "errors": errors,
        "log_likelihood": result.log_likelihood,
    }


@pytest.mark.parametrize(
    "regime",
    ["low", "moderate", "high"],
)
def test_merton_recovery_diagnostics(regime):
    experiment = run_recovery_experiment(
        regime=regime,
        n_paths=300,
        seed=42,
    )

    assert experiment["converged"]
    assert np.isfinite(
        experiment["log_likelihood"]
    )

    assert experiment["n_observations"] == 3600

    for name in PARAMETER_NAMES:
        assert np.isfinite(
            experiment["estimated"][name]
        )

    print(
        f"\nRegime: {regime}"
    )

    print(
        f"{'Parameter':<15}"
        f"{'True':>12}"
        f"{'Estimated':>12}"
        f"{'Error':>12}"
    )

    for name in PARAMETER_NAMES:
        print(
            f"{name:<15}"
            f"{experiment['true'][name]:>12.5f}"
            f"{experiment['estimated'][name]:>12.5f}"
            f"{experiment['errors'][name]:>12.5f}"
        )


def summarize_recovery(regime, n_paths, seeds):
    experiments = [
        run_recovery_experiment(
            regime=regime,
            n_paths=n_paths,
            seed=seed,
        )
        for seed in seeds
    ]

    print(
        f"\nRegime: {regime}, "
        f"Paths: {n_paths}, "
        f"Seeds: {len(seeds)}"
    )

    convergence_rate = np.mean([
        experiment["converged"]
        for experiment in experiments
    ])

    print(
        f"Convergence rate: "
        f"{convergence_rate:.1%}"
    )

    print(
        f"{'Parameter':<15}"
        f"{'Bias':>12}"
        f"{'RMSE':>12}"
    )

    for name in PARAMETER_NAMES:
        errors = np.array([
            experiment["errors"][name]
            for experiment in experiments
        ])

        bias = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))

        print(
            f"{name:<15}"
            f"{bias:>12.5f}"
            f"{rmse:>12.5f}"
        )

    return experiments


@pytest.mark.parametrize(
    "regime",
    ["low", "moderate", "high"],
)
def test_merton_multiple_seed_recovery(regime):
    experiments = summarize_recovery(
        regime=regime,
        n_paths=300,
        seeds=[10, 20, 30, 40, 50],
    )

    assert len(experiments) == 5

    for experiment in experiments:
        assert np.isfinite(
            experiment["log_likelihood"]
        )

        for name in PARAMETER_NAMES:
            assert np.isfinite(
                experiment["estimated"][name]
            )

@pytest.mark.parametrize(
    "regime",
    ["low", "moderate", "high"],
)
def test_merton_sample_size_sensitivity(regime):
    path_counts = [100, 300, 1000]
    seeds = [10, 20, 30]

    print(f"\nSample-size study: {regime}")

    print(
        f"{'Paths':>8}"
        f"{'Observations':>15}"
        f"{'Lambda RMSE':>15}"
        f"{'Sigma RMSE':>15}"
        f"{'Convergence':>15}"
    )

    for n_paths in path_counts:
        experiments = [
            run_recovery_experiment(
                regime=regime,
                n_paths=n_paths,
                seed=seed,
            )
            for seed in seeds
        ]

        lambda_errors = np.array([
            exp["errors"]["jump_lambda"]
            for exp in experiments
        ])

        sigma_errors = np.array([
            exp["errors"]["sigma"]
            for exp in experiments
        ])

        lambda_rmse = np.sqrt(
            np.mean(lambda_errors**2)
        )

        sigma_rmse = np.sqrt(
            np.mean(sigma_errors**2)
        )

        convergence = np.mean([
            exp["converged"]
            for exp in experiments
        ])

        n_observations = 12 * n_paths

        print(
            f"{n_paths:>8}"
            f"{n_observations:>15}"
            f"{lambda_rmse:>15.5f}"
            f"{sigma_rmse:>15.5f}"
            f"{convergence:>14.1%}"
        )

        assert np.isfinite(lambda_rmse)
        assert np.isfinite(sigma_rmse)

def test_high_jump_individual_estimates():
    regime = "high"
    n_paths = 300
    seeds = [10, 20, 30]

    true_parameters = REGIMES[regime]

    print("\nHigh-jump parameter diagnostics")
    print(f"True parameters: {true_parameters}")

    for seed in seeds:
        experiment = run_recovery_experiment(
            regime=regime,
            n_paths=n_paths,
            seed=seed,
        )

        print(f"\nSeed: {seed}")
        print(
            f"Converged: {experiment['converged']}"
        )
        print(
            f"Log-likelihood: "
            f"{experiment['log_likelihood']:.5f}"
        )

        for name in PARAMETER_NAMES:
            print(
                f"{name:<15}"
                f"True: {experiment['true'][name]:>9.5f}"
                f"  Estimated: "
                f"{experiment['estimated'][name]:>9.5f}"
            )

        assert experiment["converged"]
        assert np.isfinite(
            experiment["log_likelihood"]
        )

def test_high_jump_seed20_likelihood_comparison():
    true_parameters = REGIMES["high"]

    model = MertonAssetPriceModel(
        x0=100.0,
        **true_parameters,
    )

    dataset = SyntheticDataGenerator.generate(
        model,
        SimulationConfig(
            horizon=1.0,
            steps=12,
            paths=300,
            seed=20,
        ),
    )

    result = MertonCalibrator.fit(
        dataset.log_returns,
        dt=dataset.dt,
        n_starts=3,
        seed=20,
        maxiter=400,
    )

    true_ll = MertonCalibrator.log_likelihood(
        dataset.log_returns,
        dt=dataset.dt,
        **true_parameters,
    )

    fitted_ll = MertonCalibrator.log_likelihood(
        dataset.log_returns,
        dt=dataset.dt,
        **result.parameters,
    )

    print("\nSeed 20 likelihood comparison")
    print(f"True parameters LL:   {true_ll:.6f}")
    print(f"Fitted parameters LL: {fitted_ll:.6f}")
    print(f"Difference:           {fitted_ll - true_ll:.6f}")

    assert np.isfinite(true_ll)
    assert np.isfinite(fitted_ll)


def test_high_jump_seed20_multistart():
    true_parameters = REGIMES["high"]

    model = MertonAssetPriceModel(
        x0=100.0,
        **true_parameters,
    )

    dataset = SyntheticDataGenerator.generate(
        model,
        SimulationConfig(
            horizon=1.0,
            steps=12,
            paths=300,
            seed=20,
        ),
    )

    true_ll = MertonCalibrator.log_likelihood(
        dataset.log_returns,
        dt=dataset.dt,
        **true_parameters,
    )

    print("\nSeed 20 multistart diagnostic")
    print(f"True parameter LL: {true_ll:.6f}")

    results = {}

    for n_starts in (3, 10, 20):
        result = MertonCalibrator.fit(
            dataset.log_returns,
            dt=dataset.dt,
            n_starts=n_starts,
            seed=20,
            maxiter=400,
        )

        results[n_starts] = result

        print(
            f"\nStarts: {n_starts}"
            f"\nConverged: {result.converged}"
            f"\nFitted LL: {result.log_likelihood:.6f}"
            f"\nDifference from true: "
            f"{result.log_likelihood - true_ll:.6f}"
        )

        for name in PARAMETER_NAMES:
            print(
                f"  {name:<15}"
                f"{result.parameters[name]:.6f}"
            )

        assert np.isfinite(result.log_likelihood)

    assert (
        results[10].log_likelihood
        >= results[3].log_likelihood - 1e-6
    )

    assert (
        results[20].log_likelihood
        >= results[10].log_likelihood - 1e-6
    )

def test_high_jump_initialization_regression():
    true_parameters = REGIMES["high"]

    model = MertonAssetPriceModel(
        x0=100.0,
        **true_parameters,
    )

    dataset = SyntheticDataGenerator.generate(
        model,
        SimulationConfig(
            horizon=1.0,
            steps=12,
            paths=300,
            seed=20,
        ),
    )

    result = MertonCalibrator.fit(
        dataset.log_returns,
        dt=dataset.dt,
        n_starts=3,
        seed=20,
        maxiter=400,
    )

    true_ll = MertonCalibrator.log_likelihood(
        dataset.log_returns,
        dt=dataset.dt,
        **true_parameters,
    )

    assert result.converged
    assert np.isfinite(result.log_likelihood)
    assert result.log_likelihood >= true_ll - 1e-5


from typing import Callable
import numpy as np


class MonteCarloIntegrator:
    """Monte Carlo integration and variance-reduction methods."""

    @staticmethod
    def _validate_n_samples(n_samples: int) -> None:
        if (
            isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, (int, np.integer))
            or n_samples < 2
        ):
            raise ValueError("n_samples must be an integer >= 2.")

    @staticmethod
    def _validate_interval(a: float, b: float) -> None:
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("Integration bounds must be finite.")
        if a >= b:
            raise ValueError("Require a < b.")

    @staticmethod
    def _evaluate(func, x):
        """Evaluate a vectorized function and validate its output."""
        y = np.asarray(func(x), dtype=float)

        if y.shape != x.shape:
            raise ValueError(
                f"Function output must have shape {x.shape}; "
                f"received {y.shape}."
            )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "Function evaluation contains non-finite values."
            )

        return y

    @staticmethod
    def _estimate_and_se(values, scale=1.0):
        n = len(values)

        estimate = scale * np.mean(values)
        se = abs(scale) * np.std(
            values, ddof=1
        ) / np.sqrt(n)

        if not np.isfinite(estimate) or not np.isfinite(se):
            raise FloatingPointError(
                "Non-finite Monte Carlo estimate or standard error."
            )

        return float(estimate), float(se)

    @staticmethod
    def integrate_1d(
        func: Callable,
        a: float,
        b: float,
        n_samples: int = 100_000,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Estimate integral_a^b f(x) dx by uniform sampling."""
        MonteCarloIntegrator._validate_n_samples(n_samples)
        MonteCarloIntegrator._validate_interval(a, b)

        rng = np.random.default_rng(seed)
        x = rng.uniform(a, b, size=n_samples)

        y = MonteCarloIntegrator._evaluate(func, x)

        return MonteCarloIntegrator._estimate_and_se(
            y, scale=b - a
        )

    @staticmethod
    def integrate_antithetic(
        func: Callable,
        a: float,
        b: float,
        n_samples: int = 100_000,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Antithetic integration with exactly n_samples evaluations.

        n_samples must be even and at least 4.
        """
        MonteCarloIntegrator._validate_n_samples(n_samples)
        MonteCarloIntegrator._validate_interval(a, b)

        if n_samples < 4 or n_samples % 2 != 0:
            raise ValueError(
                "Antithetic sampling requires an even "
                "n_samples >= 4."
            )

        n_pairs = n_samples // 2
        rng = np.random.default_rng(seed)

        x1 = rng.uniform(a, b, size=n_pairs)
        x2 = a + b - x1

        y1 = MonteCarloIntegrator._evaluate(func, x1)
        y2 = MonteCarloIntegrator._evaluate(func, x2)

        pair_averages = 0.5 * (y1 + y2)

        # Independent units are pairs, not individual draws.
        return MonteCarloIntegrator._estimate_and_se(
            pair_averages, scale=b - a
        )

    @staticmethod
    def importance_sampling(
        target_func: Callable,
        proposal_sample_func: Callable,
        proposal_pdf_func: Callable,
        n_samples: int = 100_000,
    ) -> tuple[float, float]:
        """Estimate integral f(x) dx by importance sampling.
        """

        MonteCarloIntegrator._validate_n_samples(n_samples)

        x = np.asarray(
            proposal_sample_func(n_samples), dtype=float
        )

        if x.shape != (n_samples,):
            raise ValueError(
                "Proposal sampler must return shape (n_samples,)."
            )

        if not np.all(np.isfinite(x)):
            raise ValueError("Proposal draws must be finite.")

        f = MonteCarloIntegrator._evaluate(target_func, x)
        q = MonteCarloIntegrator._evaluate(
            proposal_pdf_func, x
        )

        if np.any(q <= 0):
            raise ValueError(
                "Proposal density must be positive at sampled points."
            )

        with np.errstate(over="ignore", divide="ignore"):
            weights = f / q

        if not np.all(np.isfinite(weights)):
            raise FloatingPointError(
                "Importance weights are non-finite."
            )

        return MonteCarloIntegrator._estimate_and_se(weights)

    @staticmethod
    def expectation_importance_sampling(
        func: Callable,
        target_pdf_func: Callable,
        proposal_sample_func: Callable,
        proposal_pdf_func: Callable,
        n_samples: int = 100_000,
    ) -> tuple[float, float]:
        """Estimate E_p[g(X)] using draws from proposal q.
        """

        def weighted_integrand(x):
            return (
                np.asarray(func(x), dtype=float)
                * np.asarray(target_pdf_func(x), dtype=float)
            )

        return MonteCarloIntegrator.importance_sampling(
            target_func=weighted_integrand,
            proposal_sample_func=proposal_sample_func,
            proposal_pdf_func=proposal_pdf_func,
            n_samples=n_samples,
        )

from typing import Callable, Optional, Tuple
import numpy as np


class MonteCarloIntegrator:
    """Monte Carlo numerical integration and variance reduction."""

    @staticmethod
    def integrate_1d(
        func: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        n_samples: int = 100_000,
    ) -> Tuple[float, float]:
        """Standard 1D Monte Carlo integration over interval [a, b].

        Returns:
            Tuple of (estimated_integral, standard_error)
        """
        x_draws = np.random.uniform(a, b, size=n_samples)
        y_evals = func(x_draws)

        integral_estimate = (b - a) * np.mean(y_evals)
        variance = np.var(y_evals, ddof=1)
        standard_error = (b - a) * np.sqrt(variance / n_samples)

        return float(integral_estimate), float(standard_error)

    @staticmethod
    def integrate_antithetic(
        func: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        n_samples: int = 100_000,
    ) -> Tuple[float, float]:
        """Monte Carlo integration using antithetic variables for variance reduction over [a, b].

        Returns:
            Tuple of (estimated_integral, standard_error)
        """
        half_samples = n_samples // 2
        u = np.random.uniform(0, 1, size=half_samples)

        x1 = a + u * (b - a)
        x2 = a + (1 - u) * (b - a)

        pair_evals = 0.5 * (func(x1) + func(x2))

        integral_estimate = (b - a) * np.mean(pair_evals)
        variance = np.var(pair_evals, ddof=1)
        standard_error = (b - a) * np.sqrt(variance / half_samples)

        return float(integral_estimate), float(standard_error)

    @staticmethod
    def importance_sampling(
        target_func: Callable[[np.ndarray], np.ndarray],
        proposal_sample_func: Callable[[int], np.ndarray],
        proposal_pdf_func: Callable[[np.ndarray], np.ndarray],
        n_samples: int = 100_000,
    ) -> Tuple[float, float]:
        """Importance sampling Monte Carlo for unbounded target expectations.

        Returns:
            Tuple of (estimated_expectation, standard_error)
        """
        x_draws = proposal_sample_func(n_samples)
        weights = target_func(x_draws) / proposal_pdf_func(x_draws)

        estimate = np.mean(weights)
        variance = np.var(weights, ddof=1)
        standard_error = np.sqrt(variance / n_samples)

        return float(estimate), float(standard_error)
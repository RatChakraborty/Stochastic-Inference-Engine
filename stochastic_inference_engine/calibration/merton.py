import numpy as np

from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import poisson

from .gbm import GBMCalibrator
from .results import CalibrationResult


class MertonCalibrator:
    """Maximum likelihood calibration of Merton jump diffusion."""

    PARAMETER_NAMES = (
        "mu",
        "sigma",
        "jump_lambda",
        "jump_mu",
        "jump_sigma",
    )

    DEFAULT_BOUNDS = (
        (-2.0, 2.0),       # mu
        (0.001, 2.0),      # sigma
        (0.0, 100.0),      # jump_lambda
        (-1.0, 1.0),       # jump_mu
        (0.0, 2.0),        # jump_sigma
    )

    @staticmethod
    def _validate_parameters(
        mu,
        sigma,
        jump_lambda,
        jump_mu,
        jump_sigma,
    ):
        values = np.asarray(
            [
                mu,
                sigma,
                jump_lambda,
                jump_mu,
                jump_sigma,
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "All model parameters must be finite."
            )

        if sigma <= 0:
            raise ValueError(
                "Diffusion volatility must be positive."
            )

        if jump_lambda < 0:
            raise ValueError(
                "Jump intensity cannot be negative."
            )

        if jump_sigma < 0:
            raise ValueError(
                "Jump volatility cannot be negative."
            )

        return values

    @staticmethod
    def _component_count(
        jump_lambda,
        dt,
        tail_tolerance,
        max_components,
    ):
        """Select enough Poisson terms to control omitted mass."""

        mean_count = jump_lambda * dt

        if mean_count == 0:
            return 1

        # The Poisson inverse survival function identifies
        # a cutoff with sufficiently small upper-tail mass.
        cutoff = poisson.isf(
            tail_tolerance, mean_count
        )

        if not np.isfinite(cutoff):
            raise ValueError(
                "Could not determine Poisson truncation."
            )

        n_components = max(
            1, int(cutoff) + 1
        )

        if n_components > max_components:
            raise ValueError(
                "Poisson truncation requires more than "
                "max_components. Increase the limit."
            )

        return n_components

    @staticmethod
    def log_likelihood(
            log_returns,
            dt,
            mu,
            sigma,
            jump_lambda,
            jump_mu,
            jump_sigma,
            tail_tolerance=1e-12,
            max_components=1000,
            chunk_size=4096,
            *,
            _fixed_components=None,
    ) -> float:
        """Compute the Merton log-return log-likelihood.
        """

        returns, dt = GBMCalibrator._validate_returns(
            log_returns, dt
        )

        MertonCalibrator._validate_parameters(
            mu,
            sigma,
            jump_lambda,
            jump_mu,
            jump_sigma,
        )

        if (
                isinstance(tail_tolerance, (bool, np.bool_))
                or not np.isscalar(tail_tolerance)
        ):
            raise ValueError(
                "tail_tolerance must be a finite number "
                "strictly between 0 and 1."
            )

        try:
            tail_tolerance = float(tail_tolerance)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "tail_tolerance must be a finite number "
                "strictly between 0 and 1."
            ) from None

        if (
                not np.isfinite(tail_tolerance)
                or not 0 < tail_tolerance < 1
        ):
            raise ValueError(
                "tail_tolerance must be strictly between 0 and 1."
            )

        for name, value in (
                ("max_components", max_components),
                ("chunk_size", chunk_size),
        ):
            if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if _fixed_components is None:
            n_components = MertonCalibrator._component_count(
                jump_lambda,
                dt,
                tail_tolerance,
                max_components,
            )
        else:
            if (
                    isinstance(_fixed_components, (bool, np.bool_))
                    or not isinstance(
                _fixed_components, (int, np.integer)
            )
                    or not 1 <= _fixed_components <= max_components
            ):
                raise ValueError(
                    "_fixed_components must be a positive integer "
                    "not exceeding max_components."
                )

            n_components = int(_fixed_components)

        k = np.arange(n_components)

        mean_count = jump_lambda * dt

        kappa = np.expm1(
            jump_mu + 0.5 * jump_sigma**2
        )

        compensated_drift = (
            mu
            - jump_lambda * kappa
            - 0.5 * sigma**2
        )

        component_means = (
            compensated_drift * dt
            + k * jump_mu
        )

        component_variances = (
            sigma**2 * dt
            + k * jump_sigma**2
        )

        log_weights = poisson.logpmf(
            k, mean_count
        )

        log_normalizers = (
            -0.5 * np.log(
                2.0 * np.pi * component_variances
            )
        )

        total = 0.0

        for start in range(0, returns.size, chunk_size):
            batch = returns[
                start:start + chunk_size
            ]

            residuals = (
                batch[:, None]
                - component_means[None, :]
            )

            log_densities = (
                log_normalizers[None, :]
                - 0.5
                * residuals**2
                / component_variances[None, :]
            )

            log_mixture = logsumexp(
                log_densities
                + log_weights[None, :],
                axis=1,
            )

            total += float(np.sum(log_mixture))

        return total

    @staticmethod
    def fit(
        log_returns,
        dt,
        n_starts=5,
        seed=42,
        bounds=None,
        maxiter=1000,
        tail_tolerance=1e-12,
    ) -> CalibrationResult:
        """Fit all five Merton parameters by bounded MLE."""

        returns, dt = GBMCalibrator._validate_returns(
            log_returns, dt
        )

        if (
            isinstance(n_starts, bool)
            or not isinstance(n_starts, int)
            or n_starts < 1
        ):
            raise ValueError(
                "n_starts must be a positive integer."
            )

        if (
            isinstance(maxiter, bool)
            or not isinstance(maxiter, int)
            or maxiter < 1
        ):
            raise ValueError(
                "maxiter must be a positive integer."
            )

        if (
                isinstance(tail_tolerance, (bool, np.bool_))
                or not np.isscalar(tail_tolerance)
        ):
            raise ValueError(
                "tail_tolerance must be strictly between 0 and 1."
            )

        try:
            tail_tolerance = float(tail_tolerance)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "tail_tolerance must be strictly between 0 and 1."
            ) from None

        if (
                not np.isfinite(tail_tolerance)
                or not 0 < tail_tolerance < 1
        ):
            raise ValueError(
                "tail_tolerance must be strictly between 0 and 1."
            )

        if bounds is None:
            bounds = MertonCalibrator.DEFAULT_BOUNDS

        bounds = np.asarray(bounds, dtype=float)

        if (
            bounds.shape != (5, 2)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError(
                "bounds must contain five valid intervals."
            )

        if (
            bounds[1, 0] <= 0
            or bounds[2, 0] < 0
            or bounds[4, 0] < 0
        ):
            raise ValueError(
                "Bounds violate parameter constraints."
            )

        rng = np.random.default_rng(seed)

        gbm_fit = GBMCalibrator.fit(returns, dt)

        gbm_mu = gbm_fit.parameters["mu"]
        gbm_sigma = gbm_fit.parameters["sigma"]

        lower = bounds[:, 0]
        upper = bounds[:, 1]

        fixed_components = MertonCalibrator._component_count(
            jump_lambda=float(upper[2]),
            dt=dt,
            tail_tolerance=tail_tolerance,
            max_components=1000,
        )

        baseline = np.array([
            gbm_mu,
            max(0.5 * gbm_sigma, 0.001),
            2.0,
            -0.05,
            max(0.5 * gbm_sigma, 0.01),
        ])

        # Alternative starting points covering different
        # jump-frequency and jump-size regimes.

        candidate_starts = [
            baseline,

            [
                gbm_mu,
                0.6 * gbm_sigma,
                12.0,
                -0.12,
                0.12,
            ],

            [
                gbm_mu,
                0.7 * gbm_sigma,
                4.0,
                -0.10,
                0.10,
            ],

            [
                gbm_mu,
                0.8 * gbm_sigma,
                0.5,
                -0.20,
                0.15,
            ],

            [
                gbm_mu,
                0.5 * gbm_sigma,
                20.0,
                -0.05,
                0.08,
            ],

            [
                gbm_mu,
                0.7 * gbm_sigma,
                6.0,
                0.0,
                0.15,
            ],

            [
                gbm_mu,
                0.7 * gbm_sigma,
                4.0,
                0.10,
                0.10,
            ],
        ]

        starting_points = []

        for candidate in candidate_starts[:n_starts]:
            starting_points.append(
                np.clip(
                    np.asarray(candidate, dtype=float),
                    lower,
                    upper,
                )
            )

        while len(starting_points) < n_starts:
            candidate = np.array([
                rng.normal(gbm_mu, 0.1),
                gbm_sigma * rng.uniform(0.3, 1.2),
                rng.uniform(
                    max(0.0, lower[2]),
                    upper[2],
                ),
                rng.uniform(-0.2, 0.2),
                gbm_sigma * rng.uniform(0.2, 1.5),
            ])

            starting_points.append(
                np.clip(candidate, lower, upper)
            )

        def negative_log_likelihood(theta):
            try:
                value = MertonCalibrator.log_likelihood(
                    returns,
                    dt,
                    *theta,
                    tail_tolerance=tail_tolerance,
                    _fixed_components=fixed_components,
                )

            except (ValueError, FloatingPointError):
                return 1e100

            if not np.isfinite(value):
                return 1e100

            return -value

        results = []

        for initial in starting_points:
            optimization = minimize(
                negative_log_likelihood,
                x0=initial,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": maxiter,
                    "ftol": 1e-10,
                },
            )

            if (
                optimization.success
                and np.isfinite(optimization.fun)
            ):
                results.append(optimization)

        if not results:
            raise RuntimeError(
                "Merton calibration did not converge "
                "from any starting point."
            )

        best = min(
            results,
            key=lambda result: result.fun,
        )

        parameters = dict(
            zip(
                MertonCalibrator.PARAMETER_NAMES,
                map(float, best.x),
            )
        )

        return CalibrationResult(
            model_name="MertonAssetPriceModel",
            parameters=parameters,
            log_likelihood=float(-best.fun),
            n_observations=returns.size,
            converged=True,
            method="Multi-start L-BFGS-B maximum likelihood",
            message=(
                f"Converged from {len(results)} "
                f"of {n_starts} starting points."
            ),
        )
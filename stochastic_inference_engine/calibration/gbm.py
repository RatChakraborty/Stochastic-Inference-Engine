import numpy as np
from .results import CalibrationResult

class GBMCalibrator:
    """Maximum likelihood calibration of GBM from log returns."""

    @staticmethod
    def _validate_returns(log_returns, dt):
        if (
            isinstance(dt, (bool, np.bool_))
            or not np.isscalar(dt)
        ):
            raise ValueError(
                "dt must be a positive finite scalar."
            )

        try:
            dt = float(dt)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "dt must be a positive finite scalar."
            ) from None

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(
                "dt must be a positive finite scalar."
            )

        try:
            returns = np.asarray(
                log_returns, dtype=float
            )
        except (TypeError, ValueError):
            raise ValueError(
                "log_returns must contain numeric values."
            ) from None

        if returns.ndim not in (1, 2):
            raise ValueError(
                "log_returns must be a 1D or 2D array."
            )

        if returns.size < 2:
            raise ValueError(
                "At least two observations are required."
            )

        if not np.all(np.isfinite(returns)):
            raise ValueError(
                "All log returns must be finite."
            )

        return returns.ravel(), dt

    @staticmethod
    def log_likelihood(
        log_returns,
        dt,
        mu,
        sigma,
    ) -> float:
        """Evaluate the GBM log-return log-likelihood."""

        returns, dt = GBMCalibrator._validate_returns(
            log_returns, dt
        )

        if not np.isfinite(mu):
            raise ValueError("mu must be finite.")

        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                "sigma must be positive and finite."
            )

        mean = (mu - 0.5 * sigma**2) * dt
        variance = sigma**2 * dt

        residuals = returns - mean

        log_likelihood = (
            -0.5 * returns.size
            * np.log(2 * np.pi * variance)
            -0.5 * np.sum(residuals**2) / variance
        )

        return float(log_likelihood)

    @staticmethod
    def fit(
        log_returns,
        dt,
    ) -> CalibrationResult:
        """Fit GBM parameters by closed-form MLE."""

        returns, dt = GBMCalibrator._validate_returns(
            log_returns, dt
        )

        n = returns.size

        mean_return = float(np.mean(returns))

        variance_return = float(
            np.var(returns, ddof=0)
        )

        if variance_return <= 0:
            raise ValueError(
                "Cannot estimate positive volatility "
                "from constant returns."
            )

        sigma_squared = variance_return / dt
        sigma_hat = float(np.sqrt(sigma_squared))

        mu_hat = float(
            mean_return / dt
            + 0.5 * sigma_squared
        )

        log_likelihood = GBMCalibrator.log_likelihood(
            returns,
            dt=dt,
            mu=mu_hat,
            sigma=sigma_hat,
        )
        sigma_se = float(
            sigma_hat / np.sqrt(2 * n)
        )

        mu_se = float(
            np.sqrt(
                sigma_squared / (n * dt)
                + sigma_squared**2 / (2 * n)
            )
        )

        return CalibrationResult(
            model_name="GeometricBrownianMotion",
            parameters={
                "mu": mu_hat,
                "sigma": sigma_hat,
            },
            log_likelihood=log_likelihood,
            n_observations=n,
            converged=True,
            method="Closed-form maximum likelihood",
            message="Analytical MLE completed.",
            standard_errors={
                "mu": mu_se,
                "sigma": sigma_se,
            },
        )

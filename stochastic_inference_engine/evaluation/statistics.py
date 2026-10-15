from dataclasses import dataclass
import numpy as np
from scipy.stats import skew, kurtosis


def validate_returns(returns, min_size=2):
    """Validate and flatten a collection of log returns."""
    try:
        values = np.asarray(returns, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "Returns must contain numeric values."
        ) from None

    if values.ndim not in (1, 2):
        raise ValueError(
            "Returns must be a 1D or 2D array."
        )

    if values.size < min_size:
        raise ValueError(
            f"At least {min_size} observations are required."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "All returns must be finite."
        )

    return values


@dataclass(frozen=True)
class ReturnStatistics:
    n_observations: int
    mean: float
    std: float
    variance: float
    skewness: float
    excess_kurtosis: float


class StatisticalEvaluator:
    """Descriptive statistics for financial log returns."""

    @staticmethod
    def summarize(returns) -> ReturnStatistics:
        values = validate_returns(returns).ravel()

        return ReturnStatistics(
            n_observations=values.size,
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=1)),
            variance=float(np.var(values, ddof=1)),
            skewness=float(skew(values, bias=False))
                if values.size >= 3
                and np.std(values) > 0 else 0.0,
            excess_kurtosis=float(
                kurtosis(values, fisher=True, bias=False)
            ) if values.size >= 4
            and np.std(values) > 0 else 0.0,
        )

    @staticmethod
    def autocorrelation(returns, lag=1) -> float:
        """Autocorrelation within a single return series.
        """
        values = validate_returns(returns)

        if values.ndim == 1:
            values = values[:, None]

        if (
            isinstance(lag, (bool, np.bool_))
            or not isinstance(lag, (int, np.integer))
            or lag < 1
            or lag >= values.shape[0]
        ):
            raise ValueError(
                "lag must be between 1 and steps - 1."
            )

        # Center each path separately.
        centered = values - np.mean(
            values, axis=0, keepdims=True
        )

        numerator = np.sum(
            centered[:-lag] * centered[lag:]
        )

        denominator = np.sum(centered**2)

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

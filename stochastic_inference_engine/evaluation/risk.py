import numpy as np
from .statistics import validate_returns


class RiskEvaluator:
    """Historical risk metrics for returns and price paths."""

    @staticmethod
    def value_at_risk(
        log_returns,
        confidence=0.95,
    ) -> float:
        """Historical log-loss VaR at the given confidence."""
        values = validate_returns(log_returns).ravel()

        if not 0 < confidence < 1:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        losses = -values

        return float(
            np.quantile(losses, confidence)
        )

    @staticmethod
    def expected_shortfall(
        log_returns,
        confidence=0.95,
    ) -> float:
        """Empirical upper-tail expected log loss.
        """
        values = validate_returns(log_returns).ravel()

        if not 0 < confidence < 1:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        losses = np.sort(-values)[::-1]
        tail_count = (1 - confidence) * losses.size

        full_count = int(np.floor(tail_count))
        fractional = tail_count - full_count

        tail_sum = float(
            np.sum(losses[:full_count])
        )

        if full_count < losses.size:
            tail_sum += (
                fractional * losses[full_count]
            )

        return float(tail_sum / tail_count)

    @staticmethod
    def maximum_drawdown(prices) -> np.ndarray:
        """Maximum fractional drawdown for each price path.
        """
        values = np.asarray(prices, dtype=float)

        if values.ndim == 1:
            values = values[:, None]

        if (
            values.ndim != 2
            or values.shape[0] < 2
            or values.shape[1] < 1
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(
                "prices must be a finite, positive "
                "time-by-path array with at least two rows."
            )

        running_max = np.maximum.accumulate(
            values, axis=0
        )

        drawdowns = (
            1.0 - values / running_max
        )

        return np.max(drawdowns, axis=0)

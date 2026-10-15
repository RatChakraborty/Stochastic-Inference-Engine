from dataclasses import dataclass
from scipy.stats import ks_2samp, wasserstein_distance
from .statistics import validate_returns


@dataclass(frozen=True)
class DistributionComparison:
    ks_statistic: float
    wasserstein_distance: float


class DistributionEvaluator:
    """Compare two empirical return distributions."""

    @staticmethod
    def compare(
        reference_returns,
        candidate_returns,
    ) -> DistributionComparison:
        reference = validate_returns(
            reference_returns
        ).ravel()

        candidate = validate_returns(
            candidate_returns
        ).ravel()

        ks_result = ks_2samp(
            reference,
            candidate,
            method="auto",
        )

        return DistributionComparison(
            ks_statistic=float(
                ks_result.statistic
            ),
            wasserstein_distance=float(
                wasserstein_distance(
                    reference, candidate
                )
            ),
        )

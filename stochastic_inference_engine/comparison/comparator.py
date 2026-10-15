from dataclasses import dataclass
import math
import numpy as np

from ..calibration import (
    CalibrationResult,
    GBMCalibrator,
    MertonCalibrator,
)
from ..evaluation.statistics import validate_returns


@dataclass(frozen=True)
class ModelScore:
    model_name: str
    calibration: CalibrationResult
    test_log_likelihood: float
    test_average_log_likelihood: float

    @property
    def train_aic(self) -> float:
        return self.calibration.aic

    @property
    def train_bic(self) -> float:
        return self.calibration.bic


@dataclass(frozen=True)
class ModelFailure:
    model_name: str
    stage: str
    error_type: str
    message: str


@dataclass(frozen=True)
class ModelComparisonResult:
    scores: tuple[ModelScore, ...]
    n_train: int
    n_test: int
    dt: float
    failures: tuple[ModelFailure, ...] = ()

    @property
    def best_model(self) -> str:
        """Winner by held-out average log-likelihood."""
        if not self.scores:
            raise ValueError(
                "No successful model scores are available."
            )

        return max(
            self.scores,
            key=lambda s: s.test_average_log_likelihood,
        ).model_name

    def ranking(self) -> tuple[ModelScore, ...]:
        return tuple(
            sorted(
                self.scores,
                key=lambda s: s.test_average_log_likelihood,
                reverse=True,
            )
        )


class ModelComparator:
    """Calibrate and compare stochastic return models."""

    @staticmethod
    def compare(
        log_returns,
        dt,
        train_fraction=0.8,
        models=("gbm", "merton"),
        merton_n_starts=5,
        merton_seed=42,
        merton_maxiter=1000,
    ) -> ModelComparisonResult:
        values = validate_returns(
            log_returns, min_size=6
        )

        if values.ndim != 1:
            raise ValueError(
                "ModelComparator requires a single "
                "1D chronological return series."
            )

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

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(
                "dt must be positive and finite."
            )

        if not (
            isinstance(train_fraction, (int, float))
            and not isinstance(train_fraction, bool)
            and math.isfinite(train_fraction)
            and 0 < train_fraction < 1
        ):
            raise ValueError(
                "train_fraction must be between 0 and 1."
            )

        allowed = {"gbm", "merton"}

        if (
            not isinstance(models, (tuple, list))
            or not models
            or any(m not in allowed for m in models)
            or len(set(models)) != len(models)
        ):
            raise ValueError(
                "models must contain unique names "
                "from ('gbm', 'merton')."
            )

        split = int(values.size * train_fraction)

        if split < 2 or values.size - split < 2:
            raise ValueError(
                "Train and test sets must each contain "
                "at least two observations."
            )

        train = values[:split]
        test = values[split:]

        scores = []
        failures = []

        for name in models:
            # Stage 1: Calibration
            try:
                if name == "gbm":
                    fitted = GBMCalibrator.fit(train, dt)

                elif name == "merton":
                    fitted = MertonCalibrator.fit(
                        train,
                        dt=dt,
                        n_starts=merton_n_starts,
                        seed=merton_seed,
                        maxiter=merton_maxiter,
                    )

            except (ValueError, RuntimeError, FloatingPointError) as exc:
                failures.append(
                    ModelFailure(
                        model_name=name,
                        stage="calibration",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            # Stage 2: Held-out evaluation
            try:
                p = fitted.parameters

                if name == "gbm":
                    test_ll = GBMCalibrator.log_likelihood(
                        test,
                        dt=dt,
                        mu=p["mu"],
                        sigma=p["sigma"],
                    )

                elif name == "merton":
                    test_ll = MertonCalibrator.log_likelihood(
                        test,
                        dt=dt,
                        mu=p["mu"],
                        sigma=p["sigma"],
                        jump_lambda=p["jump_lambda"],
                        jump_mu=p["jump_mu"],
                        jump_sigma=p["jump_sigma"],
                    )

                test_ll = float(test_ll)

                if not math.isfinite(test_ll):
                    raise ValueError(
                        "Held-out log-likelihood is not finite."
                    )

            except (ValueError, RuntimeError, FloatingPointError) as exc:
                failures.append(
                    ModelFailure(
                        model_name=name,
                        stage="evaluation",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            scores.append(
                ModelScore(
                    model_name=fitted.model_name,
                    calibration=fitted,
                    test_log_likelihood=test_ll,
                    test_average_log_likelihood=(
                            test_ll / test.size
                    ),
                )
            )

        if not scores:
            details = "; ".join(
                f"{failure.model_name} "
                f"({failure.stage}): "
                f"{failure.error_type}: {failure.message}"
                for failure in failures
            )

            raise RuntimeError(
                "All requested model comparisons failed. "
                + details
            )

        return ModelComparisonResult(
            scores=tuple(scores),
            n_train=train.size,
            n_test=test.size,
            dt=dt,
            failures=tuple(failures),
        )

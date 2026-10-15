from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
import math


@dataclass(frozen=True)
class CalibrationResult:
    """Results of fitting a stochastic model."""

    model_name: str
    parameters: Mapping[str, float]
    log_likelihood: float
    n_observations: int
    converged: bool
    method: str
    message: str = ""
    standard_errors: Mapping[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not self.model_name:
            raise ValueError("model_name cannot be empty.")

        if not self.method:
            raise ValueError("method cannot be empty.")

        if (
            isinstance(self.n_observations, bool)
            or not isinstance(self.n_observations, int)
            or self.n_observations < 1
        ):
            raise ValueError(
                "n_observations must be a positive integer."
            )

        if not math.isfinite(self.log_likelihood):
            raise ValueError(
                "log_likelihood must be finite."
            )

        parameters = dict(self.parameters)
        errors = dict(self.standard_errors)

        for name, value in parameters.items():
            if not isinstance(name, str):
                raise TypeError(
                    "Parameter names must be strings."
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Parameter {name} must be finite."
                )

        for name, value in errors.items():
            if (
                not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(
                    f"Standard error for {name} is invalid."
                )

        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(parameters),
        )

        object.__setattr__(
            self,
            "standard_errors",
            MappingProxyType(errors),
        )

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return (
            2 * self.n_parameters
            - 2 * self.log_likelihood
        )

    @property
    def bic(self) -> float:
        """Bayesian information criterion."""
        return (
            math.log(self.n_observations)
            * self.n_parameters
            - 2 * self.log_likelihood
        )

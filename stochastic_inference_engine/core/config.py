
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Validated configuration for a stochastic simulation.
    """

    horizon: float
    steps: int
    paths: int
    seed: int | None = None

    def __post_init__(self):
        if not np.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError(
                "horizon must be finite and positive."
            )

        for name in ("steps", "paths"):
            value = getattr(self, name)

            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if self.seed is not None:
            if (
                isinstance(self.seed, (bool, np.bool_))
                or not isinstance(
                    self.seed, (int, np.integer)
                )
                or self.seed < 0
                or self.seed >= 2**64
            ):
                raise ValueError(
                    "seed must be None or an integer "
                    "between 0 and 2**64 - 1."
                )

    @property
    def dt(self) -> float:
        """Time-step size."""
        return self.horizon / self.steps

    @property
    def time_grid(self) -> np.ndarray:
        """Equally spaced simulation time points."""
        return np.linspace(
            0.0,
            self.horizon,
            self.steps + 1,
        )

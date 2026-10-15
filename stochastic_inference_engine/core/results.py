
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    """Container for stochastic simulation output.
    """

    time_grid: np.ndarray
    values: np.ndarray
    model_name: str

    def __post_init__(self):
        times = np.array(
            self.time_grid, dtype=float, copy=True
        )
        values = np.array(
            self.values, dtype=float, copy=True
        )

        if times.ndim != 1:
            raise ValueError("time_grid must be 1D.")

        if values.ndim != 2:
            raise ValueError("values must be a 2D array.")

        if len(times) != values.shape[0]:
            raise ValueError(
                "Time grid length must match "
                "the number of simulation rows."
            )

        if len(times) < 2 or values.shape[1] < 1:
            raise ValueError(
                "Result must contain at least two time "
                "points and one trajectory."
            )

        if not np.all(np.isfinite(times)):
            raise ValueError("Time grid must be finite.")

        if not np.all(np.diff(times) > 0):
            raise ValueError(
                "Time grid must be strictly increasing."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Simulation values must be finite."
            )

        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError(
                "model_name must be a nonempty string."
            )

        times.setflags(write=False)
        values.setflags(write=False)

        object.__setattr__(self, "time_grid", times)
        object.__setattr__(self, "values", values)

    @property
    def terminal_values(self) -> np.ndarray:
        """Values at the final simulation time."""
        return self.values[-1].copy()

    @property
    def n_paths(self) -> int:
        return self.values.shape[1]

    @property
    def n_steps(self) -> int:
        return self.values.shape[0] - 1

    def mean_path(self) -> np.ndarray:
        """Sample mean at each time point."""
        return np.mean(self.values, axis=1)

    def variance_path(self) -> np.ndarray:
        """Sample variance at each time point."""
        if self.n_paths < 2:
            raise ValueError(
                "At least two paths are needed "
                "to estimate sample variance."
            )

        return np.var(
            self.values, axis=1, ddof=1
        )

    def quantile_path(self, q: float) -> np.ndarray:
        """Sample quantile at each time point."""
        if not np.isfinite(q) or not 0 <= q <= 1:
            raise ValueError(
                "q must be between 0 and 1."
            )

        return np.quantile(
            self.values, q, axis=1
        )

    def terminal_mean(self) -> float:
        return float(np.mean(self.values[-1]))

    def terminal_standard_error(self) -> float:
        if self.n_paths < 2:
            raise ValueError(
                "At least two paths are required."
            )

        return float(
            np.std(self.values[-1], ddof=1)
            / np.sqrt(self.n_paths)
        )

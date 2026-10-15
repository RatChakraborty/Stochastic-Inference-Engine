from dataclasses import dataclass
import numpy as np

from ..core.results import SimulationResult


@dataclass(frozen=True)
class SyntheticDataset:
    """Synthetic positive asset-price paths and return statistics.

    Prices have shape (steps + 1, paths).
    Returns have shape (steps, paths).
    """

    simulation: SimulationResult
    model_parameters: dict

    def __post_init__(self):
        if not isinstance(self.simulation, SimulationResult):
            raise TypeError(
                "simulation must be a SimulationResult."
            )

        if not isinstance(self.model_parameters, dict):
            raise TypeError(
                "model_parameters must be a dictionary."
            )

        if np.any(self.simulation.values <= 0):
            raise ValueError(
                "Synthetic asset prices must be positive."
            )

        # Prevent changes to the caller's original dictionary
        # from changing the dataset metadata.
        object.__setattr__(
            self,
            "model_parameters",
            self.model_parameters.copy(),
        )

    @property
    def prices(self) -> np.ndarray:
        """Return simulated asset-price paths."""
        return self.simulation.values.copy()

    @property
    def time_grid(self) -> np.ndarray:
        return self.simulation.time_grid.copy()

    @property
    def model_name(self) -> str:
        return self.simulation.model_name

    @property
    def n_paths(self) -> int:
        return self.simulation.n_paths

    @property
    def n_steps(self) -> int:
        return self.simulation.n_steps

    @property
    def dt(self) -> float:
        return float(
            self.time_grid[-1] / self.n_steps
        )

    @property
    def log_returns(self) -> np.ndarray:
        """Logarithmic returns along the time axis."""
        prices = self.simulation.values

        return np.log(
            prices[1:] / prices[:-1]
        )

    @property
    def simple_returns(self) -> np.ndarray:
        """Simple returns along the time axis."""
        prices = self.simulation.values

        return (
            prices[1:] / prices[:-1] - 1.0
        )

    @property
    def terminal_prices(self) -> np.ndarray:
        return self.simulation.terminal_values

    def mean_log_return(self) -> float:
        """Mean of all simulated log-return increments."""
        return float(np.mean(self.log_returns))

    def variance_log_return(self) -> float:
        """Pooled sample variance of log-return increments."""
        returns = self.log_returns.ravel()

        if len(returns) < 2:
            raise ValueError(
                "At least two returns are required."
            )

        return float(np.var(returns, ddof=1))

    def annualized_volatility(
        self,
        periods_per_year: int = 252,
    ) -> float:
        """Volatility using a specified observation frequency.

        Appropriate when returns represent equally spaced
        observations and the scaling assumptions are suitable.
        """
        if (
            isinstance(periods_per_year, (bool, np.bool_))
            or not isinstance(
                periods_per_year, (int, np.integer)
            )
            or periods_per_year <= 0
        ):
            raise ValueError(
                "periods_per_year must be a positive integer."
            )

        return float(
            np.sqrt(self.variance_log_return())
            * np.sqrt(periods_per_year)
        )

    def terminal_summary(self) -> dict:
        """Summarize the terminal asset-price distribution."""
        terminal = self.terminal_prices

        return {
            "mean": float(np.mean(terminal)),
            "median": float(np.median(terminal)),
            "std": float(np.std(terminal, ddof=1))
                if len(terminal) > 1 else 0.0,
            "min": float(np.min(terminal)),
            "max": float(np.max(terminal)),
            "q05": float(np.quantile(terminal, 0.05)),
            "q95": float(np.quantile(terminal, 0.95)),
        }

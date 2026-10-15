
import numpy as np

from .config import SimulationConfig
from .results import SimulationResult
from .protocols import StochasticModel


class SimulationEngine:
    """Execute compatible stochastic models."""

    @staticmethod
    def run(
        model: StochasticModel,
        config: SimulationConfig,
    ) -> SimulationResult:

        if not isinstance(config, SimulationConfig):
            raise TypeError(
                "config must be a SimulationConfig."
            )

        if not isinstance(model, StochasticModel):
            raise TypeError(
                "model must implement the StochasticModel interface."
            )

        simulated = np.asarray(
            model.simulate(
                horizon=config.horizon,
                steps=config.steps,
                paths=config.paths,
                seed=config.seed,
            ),
            dtype=float,
        )

        expected_shape = (
            config.steps + 1,
            config.paths,
        )

        if simulated.shape != expected_shape:
            raise ValueError(
                f"Model returned shape {simulated.shape}; "
                f"expected {expected_shape}."
            )

        return SimulationResult(
            time_grid=config.time_grid,
            values=simulated,
            model_name=type(model).__name__,
        )

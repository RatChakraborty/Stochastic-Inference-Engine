
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class StochasticModel(Protocol):
    """Interface for scalar stochastic simulation models."""

    def simulate(
        self,
        horizon: float,
        steps: int,
        paths: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return simulated paths with shape (steps + 1, paths)."""
        ...

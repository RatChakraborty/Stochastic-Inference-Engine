from dataclasses import dataclass
import numpy as np

from ..dynamics.sdes import EulerMaruyamaSimulator


@dataclass(frozen=True)
class EulerMaruyamaModel:
    """Adapt EulerMaruyamaSimulator to the common model API."""

    simulator: EulerMaruyamaSimulator
    x0: float

    def __post_init__(self):
        if not isinstance(
            self.simulator, EulerMaruyamaSimulator
        ):
            raise TypeError(
                "simulator must be an EulerMaruyamaSimulator."
            )

        if not np.isfinite(self.x0):
            raise ValueError("x0 must be finite.")

    def simulate(
        self,
        horizon: float,
        steps: int,
        paths: int,
        seed: int | None = None,
    ) -> np.ndarray:
        return self.simulator.simulate(
            x0=self.x0,
            horizon=horizon,
            steps=steps,
            paths=paths,
            seed=seed,
        )

from ..core.config import SimulationConfig
from ..core.engine import SimulationEngine
from ..dynamics.sdes import (
    GeometricBrownianMotion,
    MertonAssetPriceModel,
)
from .dataset import SyntheticDataset

class SyntheticDataGenerator:
    """Generate synthetic financial datasets from asset-price models."""

    SUPPORTED_MODELS = (
        GeometricBrownianMotion,
        MertonAssetPriceModel,
    )

    @staticmethod
    def generate(
        model,
        config: SimulationConfig,
    ) -> SyntheticDataset:
        """Simulate asset prices and construct a dataset."""

        if not isinstance(
            model, SyntheticDataGenerator.SUPPORTED_MODELS
        ):
            raise TypeError(
                "Synthetic financial data generation currently "
                "supports GeometricBrownianMotion and "
                "MertonAssetPriceModel."
            )

        if not isinstance(config, SimulationConfig):
            raise TypeError(
                "config must be a SimulationConfig."
            )

        simulation = SimulationEngine.run(model, config)

        if isinstance(model, GeometricBrownianMotion):
            parameters = {
                "x0": model.x0,
                "mu": model.mu,
                "sigma": model.sigma,
            }

        elif isinstance(model, MertonAssetPriceModel):
            parameters = {
                "x0": model.x0,
                "mu": model.mu,
                "sigma": model.sigma,
                "jump_lambda": model.jump_lambda,
                "jump_mu": model.jump_mu,
                "jump_sigma": model.jump_sigma,
            }

        return SyntheticDataset(
            simulation=simulation,
            model_parameters=parameters,
        )

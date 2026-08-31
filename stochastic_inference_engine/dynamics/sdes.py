from typing import Callable, Optional, Tuple
import numpy as np


class EulerMaruyamaSimulator:
    """General Euler-Maruyama simulator for continuous stochastic differential equations (SDEs). The Euler-Maruyama method is a simple way to approximate the solutions to a stochastic differential equation."""

    def __init__(
        self,
        drift_func: Callable[[np.ndarray, float], np.ndarray],
        diff_func: Callable[[np.ndarray, float], np.ndarray],
    ):
        self.drift_func = drift_func
        self.diff_func = diff_func

    def simulate(
        self, x0: float, horizon: float, steps: int, paths: int
    ) -> np.ndarray:

        """Simulates path trajectories for general drift and diffusion functions."""

        dt = horizon / steps
        x = np.zeros((steps + 1, paths))
        x[0] = x0

        for t_step in range(steps):
            t_current = t_step * dt
            current_x = x[t_step]
            dw = np.random.standard_normal(paths) * np.sqrt(dt)

            mu = self.drift_func(current_x, t_current)
            sigma = self.diff_func(current_x, t_current)

            x[t_step + 1] = current_x + mu * dt + sigma * dw

        return x


class GeometricBrownianMotion:
    """Geometric Brownian Motion (GBM) path simulator"""

    def __init__(self, x0: float, mu: float, sigma: float):
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma

    def simulate(self, horizon: float, steps: int, paths: int) -> np.ndarray:
        """Simulates exact log-normal state paths for GBM."""
        dt = horizon / steps
        noise = np.random.standard_normal((steps, paths))

        drift_part = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion_part = self.sigma * np.sqrt(dt) * noise
        log_increments = drift_part + diffusion_part

        total_log_change = np.cumsum(log_increments, axis=0)
        state_paths = self.x0 * np.exp(
            np.vstack([np.zeros(paths), total_log_change])
        )

        return state_paths


class MertonJumpDiffusion:
    """Merton Jump-Diffusion SDE simulator."""

    def __init__(
        self,
        mu_func: Callable[[np.ndarray, float], np.ndarray],
        sigma_func: Callable[[np.ndarray, float], np.ndarray],
        jump_lambda: float,
        jump_mu: float,
        jump_sigma: float,
    ):
        self.mu_func = mu_func
        self.sigma_func = sigma_func
        self.jump_lambda = jump_lambda
        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma

    def simulate(
        self, x0: float, horizon: float, steps: int, paths: int
    ) -> np.ndarray:
        """Simulates path trajectories containing normal diffusion and Poisson jump events."""
        dt = horizon / steps
        x = np.zeros((steps + 1, paths))
        x[0] = x0

        for t_step in range(steps):
            t_current = t_step * dt
            current_x = x[t_step]

            dw = np.random.standard_normal(paths) * np.sqrt(dt)
            mu = self.mu_func(current_x, t_current)
            sigma = self.sigma_func(current_x, t_current)

            x_next = current_x + mu * dt + sigma * dw

            has_jump = np.random.rand(paths) < (self.jump_lambda * dt)
            jump_sizes = np.random.normal(
                self.jump_mu, self.jump_sigma, size=paths
            )

            x[t_step + 1] = x_next + (has_jump * jump_sizes)

        return x
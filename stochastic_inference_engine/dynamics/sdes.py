from typing import Callable, Optional, Tuple
import numpy as np

class EulerMaruyamaSimulator:
    """Euler-Maruyama solver for scalar Ito SDEs.
    dX_t = a(X_t, t) dt + b(X_t, t) dW_t
    """

    def __init__(self, drift_func, diff_func):
        if not callable(drift_func):
            raise TypeError("drift_func must be callable.")

        if not callable(diff_func):
            raise TypeError("diff_func must be callable.")

        self.drift_func = drift_func
        self.diff_func = diff_func

    def simulate(
        self,
        x0: float,
        horizon: float,
        steps: int,
        paths: int,
        seed: int | None = None,
        brownian_increments: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulates scalar SDE paths.

        Here x0 is the initial condition,
        horizon is the duration of the simulation,
        steps is the number of time steps,
        paths is the number of simulated trajectories,
        seed is used for reproducible random number generation,
        and brownian_increments is an optional entry of Brownian increments.
        """

        if not np.isfinite(x0):
            raise ValueError("x0 must be finite.")

        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError(
                "horizon must be finite and positive."
            )

        if (
            isinstance(steps, (bool, np.bool_))
            or not isinstance(steps, (int, np.integer))
            or steps <= 0
        ):
            raise ValueError(
                "steps must be a positive integer."
            )

        if (
            isinstance(paths, (bool, np.bool_))
            or not isinstance(paths, (int, np.integer))
            or paths <= 0
        ):
            raise ValueError(
                "paths must be a positive integer."
            )

        if seed is not None and brownian_increments is not None:
            raise ValueError(
                "Provide either seed or brownian_increments, "
                "not both."
            )

        dt = horizon / steps

        if brownian_increments is None:
            rng = np.random.default_rng(seed)
            dw = rng.standard_normal(
                (steps, paths)
            ) * np.sqrt(dt)
        else:
            dw = np.asarray(
                brownian_increments, dtype=float
            )

            if dw.shape != (steps, paths):
                raise ValueError(
                    "brownian_increments must have shape "
                    f"({steps}, {paths})."
                )

            if not np.all(np.isfinite(dw)):
                raise ValueError(
                    "Brownian increments must be finite."
                )

        x = np.empty((steps + 1, paths), dtype=float)
        x[0] = x0

        for n in range(steps):
            t = n * dt
            current_x = x[n]

            drift = np.asarray(
                self.drift_func(current_x, t),
                dtype=float
            )

            diffusion = np.asarray(
                self.diff_func(current_x, t),
                dtype=float
            )

            try:
                drift = np.broadcast_to(drift, (paths,))
                diffusion = np.broadcast_to(
                    diffusion, (paths,)
                )
            except ValueError as exc:
                raise ValueError(
                    "Drift and diffusion must return "
                    "broadcast-compatible values."
                ) from exc

            if (
                not np.all(np.isfinite(drift))
                or not np.all(np.isfinite(diffusion))
            ):
                raise FloatingPointError(
                    "Non-finite drift or diffusion."
                )

            x[n + 1] = (
                current_x
                + drift * dt
                + diffusion * dw[n]
            )

            if not np.all(np.isfinite(x[n + 1])):
                raise FloatingPointError(
                    "Simulation produced non-finite values."
                )

        return x



class GeometricBrownianMotion:
    """Exact-grid simulator for geometric Brownian motion.

    dS_t = mu * S_t * dt + sigma * S_t * dW_t
    where sigma is the volatility and mu is the drift and x0 is the positive initial value.
    """


    def __init__(self, x0: float, mu: float, sigma: float):
        if not np.isfinite(x0) or x0 <= 0:
            raise ValueError("x0 must be finite and positive.")

        if not np.isfinite(mu):
            raise ValueError("mu must be finite.")

        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative.")

        self.x0 = float(x0)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def simulate(
        self,
        horizon: float,
        steps: int,
        paths: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate exact GBM observations on an equally spaced grid.
        """

        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("horizon must be finite and positive.")

        if isinstance(steps, (bool, np.bool_)) or not isinstance(
            steps, (int, np.integer)
        ) or steps <= 0:
            raise ValueError("steps must be a positive integer.")

        if isinstance(paths, (bool, np.bool_)) or not isinstance(
            paths, (int, np.integer)
        ) or paths <= 0:
            raise ValueError("paths must be a positive integer.")

        rng = np.random.default_rng(seed)

        dt = horizon / steps

        noise = rng.standard_normal((steps, paths))

        drift_part = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion_part = self.sigma * np.sqrt(dt) * noise

        log_increments = drift_part + diffusion_part

        cumulative_log_returns = np.cumsum(
            log_increments, axis=0
        )

        initial_log_returns = np.zeros((1, paths))

        all_log_returns = np.vstack([
            initial_log_returns,
            cumulative_log_returns
        ])

        with np.errstate(over="ignore", under="ignore"):
            state_paths = self.x0 * np.exp(all_log_returns)

        if not np.all(np.isfinite(state_paths)):
            raise FloatingPointError(
                "GBM simulation produced non-finite values."
            )

        if np.any(state_paths <= 0):
            raise FloatingPointError(
                "GBM simulation underflowed to a nonpositive value."
            )

        return state_paths



class MertonJumpDiffusion:
    """Euler-Maruyama simulator with additive Gaussian jumps.

    dX_t = a(X_t, t) dt + b(X_t, t) dW_t + Y dN_t

    Jump counts are Poisson distributed and individual,
    and jump sizes are independent Gaussian random variables.
    """

    def __init__(
        self,
        mu_func,
        sigma_func,
        jump_lambda: float,
        jump_mu: float,
        jump_sigma: float,
    ):
        if not callable(mu_func) or not callable(sigma_func):
            raise TypeError(
                "Drift and diffusion must be callable."
            )

        if not np.isfinite(jump_lambda) or jump_lambda < 0:
            raise ValueError(
                "jump_lambda must be finite and nonnegative."
            )

        if not np.isfinite(jump_mu):
            raise ValueError("jump_mu must be finite.")

        if not np.isfinite(jump_sigma) or jump_sigma < 0:
            raise ValueError(
                "jump_sigma must be finite and nonnegative."
            )

        self.mu_func = mu_func
        self.sigma_func = sigma_func
        self.jump_lambda = float(jump_lambda)
        self.jump_mu = float(jump_mu)
        self.jump_sigma = float(jump_sigma)

    def simulate(
        self,
        x0: float,
        horizon: float,
        steps: int,
        paths: int,
        seed: int | None = None,
    ) -> np.ndarray:

        if not np.isfinite(x0):
            raise ValueError("x0 must be finite.")

        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError(
                "horizon must be finite and positive."
            )

        for name, value in [("steps", steps), ("paths", paths)]:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        rng = np.random.default_rng(seed)
        dt = horizon / steps

        x = np.empty((steps + 1, paths), dtype=float)
        x[0] = x0

        for n in range(steps):
            t = n * dt
            current_x = x[n]

            drift = np.asarray(
                self.mu_func(current_x, t), dtype=float
            )
            diffusion = np.asarray(
                self.sigma_func(current_x, t), dtype=float
            )

            try:
                drift = np.broadcast_to(drift, (paths,))
                diffusion = np.broadcast_to(
                    diffusion, (paths,)
                )
            except ValueError as exc:
                raise ValueError(
                    "Drift and diffusion outputs must be "
                    "broadcast-compatible with paths."
                ) from exc

            if (
                not np.all(np.isfinite(drift))
                or not np.all(np.isfinite(diffusion))
            ):
                raise FloatingPointError(
                    "Non-finite drift or diffusion."
                )

            dw = rng.standard_normal(paths) * np.sqrt(dt)

            # Exact Poisson counts for this interval
            jump_counts = rng.poisson(
                self.jump_lambda * dt,
                size=paths,
            )

            # Exact aggregate of Gaussian jump sizes
            jump_increments = (
                jump_counts * self.jump_mu
                + np.sqrt(jump_counts)
                * self.jump_sigma
                * rng.standard_normal(paths)
            )

            x[n + 1] = (
                current_x
                + drift * dt
                + diffusion * dw
                + jump_increments
            )

            if not np.all(np.isfinite(x[n + 1])):
                raise FloatingPointError(
                    "Jump-diffusion produced non-finite values."
                )

        return x


class MertonAssetPriceModel:
    """Exact-grid Merton jump-diffusion asset-price simulator.

    dS/S_- = (mu - lambda*kappa)dt
             + sigma*dW + (exp(Y)-1)dN

    Y ~ Normal(jump_mu, jump_sigma**2)

    Here mu is the expected proportional growth rate.
    """

    def __init__(
        self,
        x0: float,
        mu: float,
        sigma: float,
        jump_lambda: float,
        jump_mu: float,
        jump_sigma: float,
    ):
        params = [
            x0, mu, sigma,
            jump_lambda, jump_mu, jump_sigma
        ]

        if not all(np.isfinite(v) for v in params):
            raise ValueError("All parameters must be finite.")

        if x0 <= 0:
            raise ValueError("x0 must be positive.")

        if sigma < 0 or jump_sigma < 0 or jump_lambda < 0:
            raise ValueError(
                "Volatilities and jump intensity "
                "must be nonnegative."
            )

        self.x0 = float(x0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.jump_lambda = float(jump_lambda)
        self.jump_mu = float(jump_mu)
        self.jump_sigma = float(jump_sigma)

        self.kappa = float(
            np.expm1(
                self.jump_mu + 0.5 * self.jump_sigma**2
            )
        )

        if not np.isfinite(self.kappa):
            raise ValueError(
                "Jump compensator is not finite."
            )

    def simulate(
        self,
        horizon: float,
        steps: int,
        paths: int,
        seed: int | None = None,
    ) -> np.ndarray:

        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError(
                "horizon must be finite and positive."
            )

        for name, value in [("steps", steps), ("paths", paths)]:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        rng = np.random.default_rng(seed)
        dt = horizon / steps

        x = np.empty((steps + 1, paths), dtype=float)
        x[0] = self.x0

        compensated_drift = (
            self.mu
            - self.jump_lambda * self.kappa
            - 0.5 * self.sigma**2
        )

        if not np.isfinite(compensated_drift):
            raise ValueError(
                "Compensated drift is not finite."
            )

        for n in range(steps):
            dw = rng.standard_normal(paths) * np.sqrt(dt)

            counts = rng.poisson(
                self.jump_lambda * dt,
                size=paths,
            )

            aggregate_log_jumps = (
                counts * self.jump_mu
                + np.sqrt(counts)
                * self.jump_sigma
                * rng.standard_normal(paths)
            )

            log_increment = (
                compensated_drift * dt
                + self.sigma * dw
                + aggregate_log_jumps
            )

            with np.errstate(over="ignore", under="ignore"):
                x[n + 1] = x[n] * np.exp(log_increment)

            if (
                not np.all(np.isfinite(x[n + 1]))
                or np.any(x[n + 1] <= 0)
            ):
                raise FloatingPointError(
                    "Asset-price simulation overflowed "
                    "or underflowed."
                )

        return x

    def expected_value(self, t: float) -> float:
        """Analytic expected asset price at time t."""
        if not np.isfinite(t) or t < 0:
            raise ValueError("t must be finite and nonnegative.")

        return float(self.x0 * np.exp(self.mu * t))

    def log_return_moments(self, t: float):
        """Analytic mean and variance of log(S_t / S_0)."""
        if not np.isfinite(t) or t < 0:
            raise ValueError("t must be finite and nonnegative.")

        mean = (
            self.mu
            - self.jump_lambda * self.kappa
            - 0.5 * self.sigma**2
            + self.jump_lambda * self.jump_mu
        ) * t

        variance = (
            self.sigma**2
            + self.jump_lambda
            * (self.jump_sigma**2 + self.jump_mu**2)
        ) * t

        return float(mean), float(variance)


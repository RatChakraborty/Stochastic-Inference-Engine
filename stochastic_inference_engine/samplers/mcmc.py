
from typing import Callable, Tuple
import numpy as np


def _positive_int(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")


def _stable_sigmoid(z):
    """Numerically stable logistic function."""
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    ez = np.exp(z)
    return ez / (1.0 + ez)


class Ising1DSampler:
    """1D Ising model with open boundaries, J=1, h=1.

    H(s) = -sum(s[i]*s[i+1]) - sum(s[i])

    One MCMC step updates one randomly selected spin.
    """

    def __init__(self, num_spins: int = 40):
        _positive_int(num_spins, "num_spins")
        self.m = int(num_spins)

    @staticmethod
    def _validate_run(steps, beta):
        _positive_int(steps, "steps")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError(
                "beta must be finite and nonnegative."
            )

    def _neighbor_sum(self, spins, j):
        result = 0
        if j > 0:
            result += spins[j - 1]
        if j < self.m - 1:
            result += spins[j + 1]
        return result

    def energy(self, spins):
        """Calculate the Ising Hamiltonian."""
        spins = np.asarray(spins)

        if spins.shape != (self.m,):
            raise ValueError("Invalid spin configuration shape.")

        if not np.all(np.isin(spins, (-1, 1))):
            raise ValueError("Spins must be -1 or +1.")

        return float(
            -np.sum(spins[:-1] * spins[1:])
            -np.sum(spins)
        )

    def run_metropolis(
        self,
        steps: int,
        beta: float,
        seed: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-spin Metropolis sampling."""

        self._validate_run(steps, beta)
        rng = np.random.default_rng(seed)

        spins = -np.ones(self.m, dtype=np.int8)
        states = np.empty(
            (steps + 1, self.m), dtype=np.int8
        )
        magnetization = np.empty(steps + 1)

        states[0] = spins
        magnetization[0] = np.mean(spins)

        for k in range(steps):
            j = rng.integers(self.m)

            local_field = self._neighbor_sum(spins, j) + 1
            delta_energy = 2 * spins[j] * local_field

            log_acceptance = min(
                0.0, -beta * delta_energy
            )

            if np.log(rng.random()) < log_acceptance:
                spins[j] *= -1

            states[k + 1] = spins
            magnetization[k + 1] = np.mean(spins)

        return states, magnetization

    def run_gibbs(
        self,
        steps: int,
        beta: float,
        seed: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-spin Gibbs sampling."""

        self._validate_run(steps, beta)
        rng = np.random.default_rng(seed)

        spins = -np.ones(self.m, dtype=np.int8)
        states = np.empty(
            (steps + 1, self.m), dtype=np.int8
        )
        magnetization = np.empty(steps + 1)

        states[0] = spins
        magnetization[0] = np.mean(spins)

        for k in range(steps):
            j = rng.integers(self.m)

            local_field = self._neighbor_sum(spins, j) + 1
            p_plus = _stable_sigmoid(
                2 * beta * local_field
            )

            spins[j] = (
                1 if rng.random() < p_plus else -1
            )

            states[k + 1] = spins
            magnetization[k + 1] = np.mean(spins)

        return states, magnetization

    @staticmethod
    def compute_autocovariance(
        m_history: np.ndarray,
        max_lag: int = 200,
    ) -> np.ndarray:
        """Biased autocovariance estimator, lags 0..max_lag-1.

        Uses a fixed denominator n. This convention is
        convenient for stable autocorrelation estimation.
        """
        x = np.asarray(m_history, dtype=float)

        if x.ndim != 1 or len(x) < 2:
            raise ValueError(
                "History must be a 1D array of length >= 2."
            )

        if not np.all(np.isfinite(x)):
            raise ValueError("History must be finite.")

        _positive_int(max_lag, "max_lag")

        if max_lag > len(x):
            raise ValueError(
                "max_lag cannot exceed history length."
            )

        centered = x - np.mean(x)
        n = len(x)

        return np.array([
            np.dot(
                centered[:n - lag],
                centered[lag:],
            ) / n
            for lag in range(max_lag)
        ])


class RandomWalkMetropolis2D:
    """Gaussian random-walk Metropolis for a 2D target.

    Provide either a density or a log-density.
    The target need not be normalized.
    """

    def __init__(
        self,
        target_pdf: Callable | None = None,
        *,
        target_logpdf: Callable | None = None,
    ):
        if (target_pdf is None) == (target_logpdf is None):
            raise ValueError(
                "Provide exactly one of target_pdf "
                "or target_logpdf."
            )

        if target_pdf is not None and not callable(target_pdf):
            raise TypeError("target_pdf must be callable.")

        if (
            target_logpdf is not None
            and not callable(target_logpdf)
        ):
            raise TypeError("target_logpdf must be callable.")

        self.target_pdf = target_pdf
        self.target_logpdf = target_logpdf

    def _log_density(self, state):
        x, y = float(state[0]), float(state[1])

        if self.target_logpdf is not None:
            value = float(self.target_logpdf(x, y))
            if np.isnan(value) or value == np.inf:
                raise ValueError("Invalid target log-density.")
            return value

        density = float(self.target_pdf(x, y))

        if not np.isfinite(density) or density < 0:
            raise ValueError(
                "Target density must be finite and nonnegative."
            )

        if density == 0:
            return -np.inf

        return float(np.log(density))

    def sample(
        self,
        steps: int,
        step_length: float,
        initial_state: Tuple[float, float] = (0.0, 0.0),
        seed: int | None = None,
    ) -> np.ndarray:
        """Return initial state plus steps MCMC transitions."""

        _positive_int(steps, "steps")

        if not np.isfinite(step_length) or step_length <= 0:
            raise ValueError(
                "step_length must be finite and positive."
            )

        initial = np.asarray(initial_state, dtype=float)

        if initial.shape != (2,) or not np.all(
            np.isfinite(initial)
        ):
            raise ValueError(
                "initial_state must contain two finite values."
            )

        rng = np.random.default_rng(seed)

        samples = np.empty((steps + 1, 2))
        samples[0] = initial

        current_logpdf = self._log_density(initial)

        if not np.isfinite(current_logpdf):
            raise ValueError(
                "Initial state must have positive target density."
            )

        for k in range(steps):
            current = samples[k]
            proposal = (
                current
                + step_length * rng.standard_normal(2)
            )

            if not np.all(np.isfinite(proposal)):
                raise FloatingPointError(
                    "Proposal contains non-finite values."
                )

            proposal_logpdf = self._log_density(proposal)

            log_acceptance = min(
                0.0,
                proposal_logpdf - current_logpdf,
            )

            if np.log(rng.random()) < log_acceptance:
                samples[k + 1] = proposal
                current_logpdf = proposal_logpdf
            else:
                samples[k + 1] = current

        return samples


class MCMCDiagnostics:
    """Basic diagnostics for a scalar stationary MCMC chain."""

    @staticmethod
    def autocorrelation(chain, max_lag=100):
        x = np.asarray(chain, dtype=float)

        if x.ndim != 1 or len(x) < 2:
            raise ValueError("Chain must be 1D with length >= 2.")

        if not np.all(np.isfinite(x)):
            raise ValueError("Chain must be finite.")

        _positive_int(max_lag, "max_lag")

        if max_lag >= len(x):
            raise ValueError(
                "max_lag must be smaller than chain length."
            )

        gamma = Ising1DSampler.compute_autocovariance(
            x, max_lag + 1
        )

        if gamma[0] <= 0:
            raise ValueError(
                "Autocorrelation is undefined for a constant chain."
            )

        return gamma / gamma[0]

    @staticmethod
    def effective_sample_size(chain, max_lag=100):
        """Approximate ESS using initial positive paired sums.

        Intended as a basic diagnostic, not a substitute for
        modern multi-chain convergence diagnostics.
        """
        x = np.asarray(chain, dtype=float)
        n = len(x)

        rho = MCMCDiagnostics.autocorrelation(
            x, min(max_lag, n - 1)
        )

        tau = 1.0

        # Sum adjacent autocorrelation pairs and stop
        # at the first nonpositive pair.
        for k in range(1, len(rho) - 1, 2):
            pair_sum = rho[k] + rho[k + 1]

            if pair_sum <= 0:
                break

            tau += 2 * pair_sum

        return float(n / max(1.0, tau))

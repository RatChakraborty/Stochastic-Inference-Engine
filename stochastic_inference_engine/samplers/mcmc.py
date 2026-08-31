from typing import Callable, Tuple
import numpy as np


class Ising1DSampler:
    """Markov Chain Monte Carlo samplers for the 1D Ising Spin System from Stat Phys."""

    def __init__(self, num_spins: int = 40):
        self.m = num_spins

    def run_metropolis(self, steps: int, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates 1D Ising system using the Metropolis-Hastings algorithm.

        Returns:
            Tuple of (spin_states_history, magnetization_history)
        """
        sigma = -np.ones(self.m)
        m_history = [np.mean(sigma)]
        sigma_history = [sigma.copy()]

        for _ in range(steps):
            j = np.random.randint(0, self.m)
            neighbor_sum = 0
            if j > 0:
                neighbor_sum += sigma[j - 1]
            if j < self.m - 1:
                neighbor_sum += sigma[j + 1]

            delta_h = 2 * sigma[j] * (neighbor_sum + 1)
            if np.random.rand() < np.exp(-beta * delta_h):
                sigma[j] *= -1

            m_history.append(np.mean(sigma))
            sigma_history.append(sigma.copy())

        return np.array(sigma_history), np.array(m_history)

    def run_gibbs(self, steps: int, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates 1D Ising system using Gibbs Sampling.

        Returns:
            Tuple of (spin_states_history, magnetization_history)
        """
        sigma = -np.ones(self.m)
        m_history = [np.mean(sigma)]
        sigma_history = [sigma.copy()]

        for _ in range(steps):
            j = np.random.randint(0, self.m)
            neighbor_sum = 0
            if j > 0:
                neighbor_sum += sigma[j - 1]
            if j < self.m - 1:
                neighbor_sum += sigma[j + 1]

            p_plus = 1 / (1 + np.exp(-2 * beta * (neighbor_sum + 1)))
            sigma[j] = 1 if np.random.rand() < p_plus else -1

            m_history.append(np.mean(sigma))
            sigma_history.append(sigma.copy())

        return np.array(sigma_history), np.array(m_history)

    @staticmethod
    def compute_autocovariance(
        m_history: np.ndarray, max_lag: int = 200
    ) -> np.ndarray:
        """Computes autocovariance gamma_k across lag steps for covergence analysis."""
        n_len = len(m_history)
        m_bar = np.mean(m_history)
        gamma = []
        for k in range(max_lag):
            val = np.sum(
                (m_history[: n_len - k] - m_bar) * (m_history[k:] - m_bar)
            ) / (n_len - k - 1)
            gamma.append(val)
        return np.array(gamma)


class RandomWalkMetropolis2D:
    """2D Random-Walk Metropolis sampler for arbit target density functions."""

    def __init__(self, target_pdf: Callable[[float, float], float]):
        self.target_pdf = target_pdf

    def sample(
        self,
        steps: int,
        step_length: float,
        initial_state: Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Draws samples using a Gaussian proposal distribution."""
        samples = np.zeros((steps + 1, 2))
        samples[0] = initial_state

        for k in range(steps):
            current_x = samples[k]
            z = np.random.normal(0, 1, 2)
            proposal = current_x + step_length * z

            f_w = self.target_pdf(proposal[0], proposal[1])
            f_xk = self.target_pdf(current_x[0], current_x[1])
            alpha = min(1.0, f_w / f_xk if f_xk > 0 else 1.0)

            if np.random.rand() < alpha:
                samples[k + 1] = proposal
            else:
                samples[k + 1] = current_x

        return samples
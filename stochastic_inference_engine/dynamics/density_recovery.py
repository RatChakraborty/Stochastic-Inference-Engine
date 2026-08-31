from typing import Tuple
import numpy as np
from scipy.stats import lognorm, norm


class FFTDensityRecovery:
    """Density recovery engine using Fast Fourier Transform (FFT) inversion on characteristic functions."""

    @staticmethod
    def recover_gbm(
        x0: float,
        T: float,
        mu: float,
        sigma: float,
        grid_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Recovers log-normal density for GBMs via FFT inversion.

        Returns:
            Tuple containing (grid_values, recovered_density, true_density, mean_squared_error)
        """
        m_drift = (mu - 0.5 * sigma**2) * T
        v_var = sigma**2 * T

        x_width = 10 * np.sqrt(v_var)
        dx = x_width / grid_size
        x_grid = np.arange(grid_size) * dx + (np.log(x0) + m_drift - x_width / 2)

        du = 2 * np.pi / x_width
        u = (np.arange(grid_size) - grid_size / 2) * du

        phi = np.exp(1j * u * (np.log(x0) + m_drift) - 0.5 * v_var * u**2)
        phi_shifted = np.fft.ifftshift(phi * np.exp(-1j * u * x_grid[0]))

        recovered_log_density = np.fft.ifft(phi_shifted).real / dx
        values = np.exp(x_grid)
        recovered_density = recovered_log_density / values

        s_param = sigma * np.sqrt(T)
        scale_param = x0 * np.exp((mu - 0.5 * sigma**2) * T)
        true_pdf = lognorm.pdf(values, s=s_param, scale=scale_param)

        mse = float(np.mean((recovered_density - true_pdf) ** 2))

        return values, recovered_density, true_pdf, mse

    @staticmethod
    def recover_general_bm(
        x0: float,
        T: float,
        sigma: float,
        grid_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Recovers Gaussian density across real space for General BM via FFT inversion.

        Returns:
            Tuple containing (x_grid, recovered_density, true_density, mean_squared_error)
        """
        x_width = 12 * sigma * np.sqrt(T)
        dx = x_width / grid_size
        x_grid = np.arange(grid_size) * dx + (x0 - x_width / 2)

        du = 2 * np.pi / x_width
        u = (np.arange(grid_size) - grid_size / 2) * du

        phi = np.exp(1j * u * x0 - 0.5 * (sigma**2 * T) * u**2)

        phi_shifted = np.fft.ifftshift(phi * np.exp(-1j * u * x_grid[0]))
        recovered_density = np.fft.ifft(phi_shifted).real / dx

        true_pdf = norm.pdf(x_grid, loc=x0, scale=sigma * np.sqrt(T))
        mse = float(np.mean((recovered_density - true_pdf) ** 2))

        return x_grid, recovered_density, true_pdf, mse
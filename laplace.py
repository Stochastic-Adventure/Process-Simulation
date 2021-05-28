"""
1D Laplace process simulation according to Madan.
"""
import numpy as np


class LaplaceProcess:
    """
    Laplace process simulation.
    """
    def __init__(self, timestep: int, mu: float, sigma: float, start_price: float):
        self.timestep = timestep
        self.mu = mu
        self.sigma = sigma
        self.start_price = start_price
        self.rng = np.random.default_rng()

    def generate_path(self, n_path: int):
        """

        :param n_path:
        :return:
        """

        rvs = self.rng.laplace(size=(n_path, self.timestep))

        dt = 1. / self.timestep

        exponent = self.mu * dt + self.sigma * np.sqrt(dt) * rvs
        exponent = np.insert(exponent, 0, 0, axis=1)
        log_returns = np.cumsum(exponent, axis=1)

        prices = self.start_price * np.exp(log_returns)

        return prices



"""
Simulate one or multi-asset geometric brownian motion.
Questionable correlation matrix is repaired in-place to maintain positive definiteness.
"""

import numpy as np
from statsmodels.stats.moment_helpers import corr2cov
from statsmodels.stats.correlation_tools import corr_nearest


class GBMProcess:
    """
    GBM simulator.
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
        rvs = self.rng.standard_normal((n_path, self.timestep))

        dt = 1. / self.timestep

        exponent = dt * (self.mu - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(dt) * rvs
        exponent = np.insert(exponent, 0, 0, axis=1)
        log_returns = np.cumsum(exponent, axis=1)

        prices = self.start_price * np.exp(log_returns)

        return prices


class CorrelatedGBM:
    """
    Correlated GBM simulator.
    """
    def __init__(self, timestep: int, mus: np.array, sigmas: np.array, corr: np.array, start_prices: np.array):
        self.n_assets = mus.shape[0]
        if corr.shape[0] != corr.shape[1]:
            raise ValueError("Error: Correlation Matrix must be a square matrix!")
        if corr.shape[0] != mus.shape[0]:
            raise ValueError("Error: Mean vector and correlation matrix mismatch!")
        if mus.shape != sigmas.shape:
            raise ValueError("Error: Mean vector and std vector shape mismatch!")
        if mus.shape != start_prices.shape:
            raise ValueError("Error: Mean vector and starting price vector shape mismatch!")

        self.mus = mus
        self.sigmas = sigmas
        # Fix the illegal correlation matrix in-place so we can use cholesky decomposition
        self.corr = corr_nearest(corr)
        self.timestep = timestep
        self.covar = corr2cov(self.corr, self.sigmas)
        self.start_prices = start_prices
        self.rng = np.random.default_rng()

    def generate_path(self, n_path: int):
        """

        :param n_path:
        :return:
        """
        dt = 1. / self.timestep

        rvs = self.rng.multivariate_normal(np.zeros((self.n_assets, )), self.covar, (n_path, self.timestep))

        exponent_1 = dt * (self.mus - 0.5 * self.sigmas * self.sigmas)
        exponent_1 = exponent_1.reshape(1, -1)
        exponent_2 = np.sqrt(dt) * rvs
        exponent = exponent_1 + exponent_2

        exponent = np.insert(exponent, 0, 0, axis=1)
        log_returns = np.cumsum(exponent, axis=1)

        prices = self.start_prices.reshape(1, -1) * np.exp(log_returns)
        return prices

    def get_paths(self, path: np.array):
        """
        Retrieve separate paths for plotting purposes.
        Path should be of shape (timestep, n_asset)
        :param path:
        :return:
        """
        return [path[:, i] for i in range(self.n_assets)]

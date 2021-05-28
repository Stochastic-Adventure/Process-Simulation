"""
Variance Gamma and NIG simulation.
"""

import numpy as np

class VarianceGamma:
    """

    """
    def __init__(self, timestep: int, mu: float, sigma: float, beta: float, start_price: float):
        self.timestep = timestep
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.start_price = start_price
        self.rng = np.random.default_rng()

    def generate_path(self, n_path: int):
        """

        :param n_path:
        :return:
        """
        dt = 1. / self.timestep

        G = self.rng.gamma(dt / self.beta, self.beta, size=(n_path, self.timestep))
        Z = self.rng.standard_normal(size=(n_path, self.timestep))

        exponent = self.mu * G + self.sigma * np.sqrt(G) * Z
        exponent = np.insert(exponent, 0, 0, axis=1)
        log_returns = np.cumsum(exponent, axis=1)

        prices = self.start_price * np.exp(log_returns)
        return prices


class NormalInverseGaussian:
    """

    """
    def __init__(self, timestep: int, delta: float, gamma: float, beta: float, start_price: float):
        self.timestep = timestep
        self.delta = delta
        self.gamma = gamma
        self.beta = beta
        self.start_price = start_price
        self.rng = np.random.default_rng()

    def generate_path(self, n_path: int):
        """

        :return:
        """
        dt = 1. / self.timestep
        self.delta *= dt

        IG = self.rng.wald(self.delta / self.gamma, self.delta ** 2, size=(n_path, self.timestep))
        exponent = self.beta * IG + self.rng.standard_normal((n_path, self.timestep)) * np.sqrt(IG)
        exponent = np.insert(exponent, 0, 0, axis=1)
        log_returns = np.cumsum(exponent, axis=1)

        prices = self.start_price * np.exp(log_returns)
        return log_returns, prices


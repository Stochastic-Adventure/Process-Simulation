"""
Jump Diffusion Simulator.
"""
import numpy as np

class MertonJumpDiffusion:
    """
    Jump diffusion model where the jump is modeled after a lognormal process.
    """
    def __init__(self, timestep: int, mu: float, sigma: float, lam: float,
                 log_a: float, log_b: float, start_price: float):
        self.timestep = timestep
        self.mu = mu
        self.sigma = sigma
        # Poisson rvs parameter
        self.lam = lam
        # Jump log-normal parameters
        self.log_a = log_a
        self.log_b = log_b
        self.start_price = start_price
        self.rng = np.random.default_rng()

    def generate_path(self, n_path: int):
        """

        :param n_path:
        :return:
        """
        dt = 1. / self.timestep

        # C.f. Glasserman p.138
        Z = self.rng.standard_normal((n_path, self.timestep))
        N = self.rng.poisson(self.lam * dt, (n_path, self.timestep))
        Z_jump = self.rng.standard_normal((n_path, self.timestep))
        M = self.log_a * N + self.log_b * np.sqrt(N) * Z_jump

        exponent = (self.mu - self.sigma ** 2 * 0.5) * dt + self.sigma * np.sqrt(dt) * Z + M

        exponent = np.insert(exponent, 0, 0, axis=1)
        log_returns = np.cumsum(exponent, axis=1)

        prices = self.start_price * np.exp(log_returns)
        return prices


class KouJumpDiffusion:
    """
    Jump diffusion model where the jump is modeled after a Gamma process.
    """
    def __init__(self, timestep: int, mu: float, sigma: float, lam: float,
                 eta_up: float, eta_down: float, q: float, start_price: float):
        self.timestep = timestep
        self.mu = mu
        self.sigma = sigma
        # Poisson rvs parameter
        self.lam = lam
        # Positive jump magnitude
        self.eta_up = eta_up
        # Negative jump magnitude
        self.eta_down = eta_down
        # Positive jump probability
        self.q = q
        self.start_price = start_price
        self.rng = np.random.default_rng()

    def random_dexp(self, n: int, q: float, eta_up: float, eta_down: float):
        """
        Double Exponential random number generator.
        :param n:
        :param q:
        :param eta_up:
        :param eta_down:
        :return:
        """
        bernoulli = self.rng.uniform(size=n) < q

        uniform = self.rng.uniform(size=n)
        dexp = np.zeros((n, ))
        dexp[bernoulli == 1] = -np.log(uniform[bernoulli == 1]) / eta_up
        dexp[bernoulli == 0] = np.log(uniform[bernoulli == 0]) / eta_down

        return dexp

    def generate_path(self, n_path: int):
        """

        :param n_path:
        :return:
        """
        dt = 1. / self.timestep

        # C.f. Glasserman p.138
        Z = self.rng.standard_normal((n_path, self.timestep))
        N = self.rng.poisson(self.lam * dt, size=(n_path, self.timestep))
        M = np.zeros((n_path, self.timestep))
        for i in range(n_path):
            for t in range(self.timestep):
                if N[i, t] > 0:
                    M[i, t] = np.sum(self.random_dexp(N[i, t], self.q, self.eta_up, self.eta_down))

        exponent = (self.mu - self.sigma ** 2 * 0.5) * dt + self.sigma * np.sqrt(dt) * Z + M

        exponent = np.insert(exponent, 0, 0, axis=1)
        log_returns = np.cumsum(exponent, axis=1)

        prices = self.start_price * np.exp(log_returns)
        return prices

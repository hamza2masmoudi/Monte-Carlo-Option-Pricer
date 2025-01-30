import numpy as np

class GeometricBrownianMotion:
    def __init__(self, S0, drift, volatility, maturity, n_steps):
        self.S0 = S0
        self.drift = drift
        self.volatility = volatility
        self.maturity = maturity
        self.n_steps = n_steps
        self.dt = maturity / n_steps

    def generate_paths(self, n_sims, antithetic=False):
        if not antithetic:
            paths = np.zeros((n_sims, self.n_steps + 1))
            paths[:, 0] = self.S0
            for t in range(1, self.n_steps+1):
                Z = np.random.normal(0, 1, n_sims)
                paths[:, t] = paths[:, t-1] * np.exp(
                    (self.drift - 0.5*self.volatility**2)*self.dt
                    + self.volatility*np.sqrt(self.dt)*Z
                )
            return paths
        else:
            # 2*n_sims with antithetic
            paths_all = np.zeros((2*n_sims, self.n_steps + 1))
            paths_all[:, 0] = self.S0
            half1 = paths_all[:n_sims, :]
            half2 = paths_all[n_sims:, :]

            for t in range(1, self.n_steps+1):
                Z = np.random.normal(0, 1, n_sims)
                half1[:, t] = half1[:, t-1] * np.exp(
                    (self.drift - 0.5*self.volatility**2)*self.dt
                    + self.volatility*np.sqrt(self.dt)*Z
                )
                half2[:, t] = half2[:, t-1] * np.exp(
                    (self.drift - 0.5*self.volatility**2)*self.dt
                    + self.volatility*np.sqrt(self.dt)*(-Z)
                )
            return paths_all
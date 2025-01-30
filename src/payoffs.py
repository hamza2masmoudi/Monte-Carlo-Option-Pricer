import numpy as np

def european_call_payoff(paths, strike):
    return np.maximum(paths[:, -1] - strike, 0.0)

def european_put_payoff(paths, strike):
    return np.maximum(strike - paths[:, -1], 0.0)

def asian_call_payoff(paths, strike):
    # Average price from t=1 to t=n_steps
    avg_price = np.mean(paths[:, 1:], axis=1)
    return np.maximum(avg_price - strike, 0.0)
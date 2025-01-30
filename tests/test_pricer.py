import pytest
import numpy as np
from scipy.stats import norm

from src.geometric_brownian_motion import GeometricBrownianMotion
from src.payoffs import european_call_payoff
from src.options_pricer import MonteCarloPricer

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    from math import erf, sqrt
    # or import from scipy.stats
    # We use norm.cdf for clarity:
    from scipy.stats import norm
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

@pytest.fixture
def european_call():
    model = GeometricBrownianMotion(S0=100, drift=0.05, volatility=0.2, maturity=1, n_steps=252)
    pricer = MonteCarloPricer(model, european_call_payoff, strike=100, r=0.05, n_sims=100000)
    return pricer

def test_european_call_price(european_call):
    mc_price, stderr = european_call.price()
    bs_price = black_scholes_call(100, 100, 1, 0.05, 0.2)
    # Tolerance of e.g. 0.5 for simulation
    assert abs(mc_price - bs_price) < 0.5
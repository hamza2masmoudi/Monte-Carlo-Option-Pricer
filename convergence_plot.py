import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.geometric_brownian_motion import GeometricBrownianMotion
from src.payoffs import european_call_payoff
from src.options_pricer import MonteCarloPricer

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def plot_convergence():
    model = GeometricBrownianMotion(100, 0.05, 0.2, 1, 252)
    pricer = MonteCarloPricer(model, european_call_payoff, strike=100, r=0.05)

    n_sims_range = np.logspace(2, 5, 20).astype(int)
    prices = []

    for n in n_sims_range:
        pricer.n_sims = n
        price, _ = pricer.price()
        prices.append(price)

    plt.figure(figsize=(10, 6))
    plt.plot(n_sims_range, prices, label='Monte Carlo Price')
    bs_val = black_scholes_call(100, 100, 1, 0.05, 0.2)
    plt.axhline(bs_val, color='r', linestyle='--', label='Black-Scholes Price')

    plt.xscale('log')
    plt.xlabel('Number of Simulations (log scale)')
    plt.ylabel('Option Price')
    plt.title('Monte Carlo Price Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence_plot.png')
    plt.show()

if __name__ == "__main__":
    plot_convergence()
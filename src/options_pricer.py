import numpy as np
import copy

class MonteCarloPricer:
    def __init__(self, model, payoff_func, **kwargs):
        """
        :param model: either a GeometricBrownianMotion instance 
                      or an ML-based generator with .generate_paths().
        :param payoff_func: function(paths, strike) -> payoff array
        :param kwargs: e.g. strike=..., r=..., n_sims=..., antithetic=...
        """
        self.model = model
        self.payoff_func = payoff_func
        self.strike = float(kwargs.get('strike', 100))
        self.r = float(kwargs.get('r', 0.05))
        self.n_sims = int(kwargs.get('n_sims', 10000))
        self.use_antithetic = bool(kwargs.get('antithetic', False))

    def price(self):
        # Generate paths:
        # If model is GBM, it may have an antithetic param:
        if hasattr(self.model, 'generate_paths'):
            try:
                paths = self.model.generate_paths(self.n_sims, antithetic=self.use_antithetic)
            except TypeError:
                # ML-based generator might not accept "antithetic" as a param
                paths = self.model.generate_paths(n_sims=self.n_sims)
        else:
            raise ValueError("Model has no generate_paths method.")

        # Compute payoffs
        payoffs = self.payoff_func(paths, self.strike)

        # Discount factor
        maturity = getattr(self.model, 'maturity', 1.0)  # default 1 if not in ML model
        discount_factor = np.exp(-self.r * maturity)

        # Price & standard error
        price = discount_factor * np.mean(payoffs)
        stderr = discount_factor * np.std(payoffs, ddof=1) / np.sqrt(len(payoffs))
        return price, stderr

def compute_greeks(pricer, bump_size=0.01):
    """
    Simple bump-and-revalue approach for Delta, Gamma, Vega (only works with GBM).
    """
    base_price, _ = pricer.price()

    S0 = getattr(pricer.model, 'S0', None)
    sigma = getattr(pricer.model, 'volatility', None)
    if S0 is None or sigma is None:
        raise ValueError("compute_greeks requires a model with S0 and volatility attributes (GBM).")

    # Bump up
    pricer_up = copy.deepcopy(pricer)
    pricer_up.model.S0 = pricer.model.S0 * (1 + bump_size)
    price_up, _ = pricer_up.price()

    # Bump down
    pricer_down = copy.deepcopy(pricer)
    pricer_down.model.S0 = pricer.model.S0 * (1 - bump_size)
    price_down, _ = pricer_down.price()

    delta = (price_up - price_down) / (pricer.model.S0 * 2 * bump_size)
    gamma = (price_up - 2*base_price + price_down) / ((pricer.model.S0 * bump_size)**2)

    # Vega
    pricer_vol_up = copy.deepcopy(pricer)
    pricer_vol_up.model.volatility = pricer.model.volatility + bump_size
    price_vol_up, _ = pricer_vol_up.price()

    vega = (price_vol_up - base_price) / bump_size

    return {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
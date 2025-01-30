import pandas as pd
from pathlib import Path

from src.geometric_brownian_motion import GeometricBrownianMotion
from src.payoffs import european_call_payoff, european_put_payoff, asian_call_payoff
from src.advanced_payoffs import (
    down_and_out_call_payoff,
    up_and_out_call_payoff,
    digital_call_payoff,
    digital_put_payoff,
    lookback_call_payoff,
    lookback_put_payoff
)
from src.ml_forecasting import MLPathGenerator
from src.options_pricer import MonteCarloPricer

def main():
    data_path = Path('data/options.csv')
    options_df = pd.read_csv(data_path)

    # Standard payoffs
    standard_map = {
        'european_call': european_call_payoff,
        'european_put': european_put_payoff,
        'asian_call': asian_call_payoff
    }

    # Advanced payoffs
    advanced_map = {
        'down_and_out_call': down_and_out_call_payoff,
        'up_and_out_call': up_and_out_call_payoff,
        'digital_call': digital_call_payoff,
        'digital_put': digital_put_payoff,
        'lookback_call': lookback_call_payoff,
        'lookback_put': lookback_put_payoff
    }

    results = []

    for _, row in options_df.iterrows():
        # Make sure we parse numeric fields properly
        S = float(row['S'])
        K = float(row['K'])
        T = float(row['T'])
        r = float(row['r'])
        sigma = float(row['sigma'])
        n_steps = int(row['n_steps'])
        n_sims = int(row['n_sims'])
        model_type = row.get('model_type', 'gbm')
        antithetic_str = str(row.get('antithetic', 'False'))
        antithetic = (antithetic_str.lower() == 'true')

        barrier = float(row.get('barrier', 0.0))
        rebate = float(row.get('rebate', 0.0))
        cash_payout = float(row.get('cash_payout', 0.0))
        ticker_symbol = str(row.get('ticker_symbol', 'None'))
        option_type = str(row['option_type'])

        # 1) Build the model
        if model_type == 'gbm':
            model = GeometricBrownianMotion(
                S0=S,
                drift=r,
                volatility=sigma,
                maturity=T,
                n_steps=n_steps
            )
        elif model_type == 'ml':
            # We'll train quickly or assume you want a short training
            gen = MLPathGenerator(ticker_symbol=ticker_symbol)
            gen.fetch_and_train(start="2022-01-01", end="2023-09-30")
            model = gen
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # 2) Select the payoff
        if option_type in standard_map:
            payoff_func = standard_map[option_type]
        elif option_type in advanced_map:
            if option_type == 'down_and_out_call':
                # Wrap payoff with barrier & rebate
                payoff_func = lambda paths, strike: down_and_out_call_payoff(paths, strike, barrier, rebate)
            elif option_type == 'up_and_out_call':
                payoff_func = lambda paths, strike: up_and_out_call_payoff(paths, strike, barrier, rebate)
            elif option_type == 'digital_call':
                payoff_func = lambda paths, strike: digital_call_payoff(paths, strike, cash_payout)
            elif option_type == 'digital_put':
                payoff_func = lambda paths, strike: digital_put_payoff(paths, strike, cash_payout)
            elif option_type == 'lookback_call':
                payoff_func = lambda paths, strike: lookback_call_payoff(paths)
            elif option_type == 'lookback_put':
                payoff_func = lambda paths, strike: lookback_put_payoff(paths)
            else:
                raise ValueError(f"Unrecognized advanced option_type: {option_type}")
        else:
            raise ValueError(f"Unknown option_type: {option_type}")

        # 3) Build pricer
        pricer = MonteCarloPricer(
            model=model,
            payoff_func=payoff_func,
            strike=K,
            r=r,
            n_sims=n_sims,
            antithetic=antithetic
        )

        # 4) Price
        price, stderr = pricer.price()

        results.append({
            'option_type': option_type,
            'model_type': model_type,
            'price': price,
            'stderr': stderr,
            'barrier': barrier,
            'rebate': rebate,
            'cash_payout': cash_payout,
            'S_used': S,
            'K_used': K,
            'T_used': T
        })

    # Print results
    results_df = pd.DataFrame(results)
    print("\nOption Pricing Results:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
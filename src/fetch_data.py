import yfinance as yf
import pandas as pd
import csv
import os

def fetch_and_append_option_data(ticker_symbol, expiry_date, option_type='european_call',
                                 model_type='gbm', barrier=0.0, rebate=0.0,
                                 cash_payout=0.0, n_steps=252, n_sims=10000,
                                 antithetic=False):
    """
    Example of pulling an option chain from yfinance. 
    Writes a row to data/options.csv with float columns 
    to avoid CSV parsing issues.
    """
    ticker = yf.Ticker(ticker_symbol)

    hist = ticker.history(period="1d")
    if hist.empty:
        raise ValueError(f"No data for {ticker_symbol}")
    last_price = float(hist['Close'].iloc[-1])

    try:
        opt_chain = ticker.option_chain(expiry_date)
    except Exception:
        raise ValueError(f"No option chain found for {ticker_symbol} on {expiry_date}.")

    if 'call' in option_type.lower():
        chain_df = opt_chain.calls
    else:
        chain_df = opt_chain.puts
    if chain_df.empty:
        raise ValueError(f"No data for {option_type} on {expiry_date}")

    # pick ATM
    chain_df['diff'] = (chain_df['strike'] - last_price).abs()
    chain_df.sort_values('diff', inplace=True)
    atm_option = chain_df.iloc[0]

    strike = float(atm_option['strike'])
    implied_vol = float(atm_option['impliedVolatility'])
    r = 0.03  # or fetch from your own source

    row_dict = {
        'option_type': option_type,
        'S': last_price,
        'K': strike,
        'T': 1.0,  # approximate
        'r': r,
        'sigma': implied_vol,
        'n_steps': n_steps,
        'n_sims': n_sims,
        'model_type': model_type,
        'antithetic': str(antithetic),
        'barrier': barrier,
        'rebate': rebate,
        'cash_payout': cash_payout,
        'ticker_symbol': ticker_symbol
    }

    csv_file = 'data/options.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

    print(f"Appended to {csv_file}:\n{row_dict}")

if __name__ == "__main__":
    # Example usage
    fetch_and_append_option_data(
        ticker_symbol="AAPL",
        expiry_date="2025-01-31",
        option_type="down_and_out_call",
        barrier=90.0,
        rebate=0.0,
        cash_payout=0.0,
        antithetic=False
    )
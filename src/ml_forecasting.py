import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class MLPathGenerator:
    """
    Example approach to train an MLP on past returns, then generate 
    future price paths by iteratively predicting next-day returns.
    """
    def __init__(self, ticker_symbol='AAPL', lookback=5, hidden_layer_sizes=(64,64)):
        self.ticker_symbol = ticker_symbol
        self.lookback = lookback
        self.hidden_layer_sizes = hidden_layer_sizes

        self.model = None
        self.scaler = None
        self.last_returns = None
        self.last_close = None

    def fetch_and_train(self, start="2020-01-01", end=None):
        # 1) Get historical data
        df = yf.download(self.ticker_symbol, start=start, end=end)
        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)

        if len(df) < self.lookback+1:
            raise ValueError("Not enough data to train ML model.")

        self.last_close = df['Close'].iloc[-1]

        # 2) Build supervised dataset
        X, y = [], []
        for i in range(self.lookback, len(df)):
            past_returns = df['Return'].iloc[i-self.lookback:i].values
            X.append(past_returns)
            y.append(df['Return'].iloc[i])
        X = np.array(X)
        y = np.array(y)

        # 3) Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 4) Train MLP
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                  activation='relu',
                                  max_iter=500,
                                  random_state=42)
        self.model.fit(X_scaled, y)

        # 5) Store the last 'lookback' returns
        self.last_returns = df['Return'].iloc[-self.lookback:].values

    def generate_paths(self, n_sims=1000, n_steps=252, initial_price=None, noise_std=0.01):
        if self.model is None:
            raise ValueError("ML model not trained. Call fetch_and_train() first.")

        if initial_price is None:
            initial_price = self.last_close

        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = initial_price

        state_matrix = np.tile(self.last_returns, (n_sims, 1))

        for t in range(1, n_steps+1):
            # Predict
            X_scaled = self.scaler.transform(state_matrix)
            predicted_returns = self.model.predict(X_scaled)

            # Add noise
            noise = np.random.normal(0, noise_std, size=n_sims)
            actual_returns = predicted_returns + noise

            # --- CLIP to [-0.5, 0.5] => max daily move of +/-50% ---
            actual_returns = np.clip(actual_returns, -0.5, 0.5)

            # Update paths
            paths[:, t] = paths[:, t-1] * (1.0 + actual_returns)

            # Roll the window
            state_matrix = np.roll(state_matrix, -1, axis=1)
            state_matrix[:, -1] = actual_returns

        return paths
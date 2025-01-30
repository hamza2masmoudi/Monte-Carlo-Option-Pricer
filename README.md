# Monte Carlo Option Pricer

This project provides a **Monte Carlo simulation** framework for pricing various option types, including **European**, **Asian**, **Digital**, **Lookback**, and **Barrier** (knock-out) options. It also integrates a simple **machine learning** module to generate underlying asset price paths as an alternative to standard Geometric Brownian Motion (GBM).

## Project Overview

1. **Geometric Brownian Motion (GBM) Model**  
   - A classic stochastic model for simulating asset prices over time.
   - Supports **antithetic sampling** for variance reduction.

2. **Machine Learning Path Generator**  
   - Trains an **MLPRegressor** on historical returns (fetched from Yahoo Finance using `yfinance`) to forecast daily returns.
   - Generates future price paths by iteratively predicting next-day returns (plus optional noise).
   - Includes a **clipping** step to prevent unrealistic explosive returns.

3. **Flexible Payoff Functions**  
   - **Standard**: European call/put, Asian call.  
   - **Advanced**: Barrier (down-and-out), digital (cash-or-nothing), lookback options.  

4. **Architecture**  
   - **`src/geometric_brownian_motion.py`**: GBM path generator.  
   - **`src/ml_forecasting.py`**: ML-based path generator (with an MLP).  
   - **`src/payoffs.py`**: Standard payoff definitions (European, Asian).  
   - **`src/advanced_payoffs.py`**: Exotic payoff definitions (barrier, digital, lookback).  
   - **`src/options_pricer.py`**: Central Monte Carlo pricer, plus a `compute_greeks` function for basic Greeks (Delta, Gamma, Vega).  
   - **`src/fetch_data.py`**: Optional script to fetch real option chain data from Yahoo Finance and append them to `data/options.csv`.  

5. **CSV Configuration**  
   - The script reads from `data/options.csv`, where each row specifies parameters like `S`, `K`, `T`, `r`, `sigma`, `barrier`, etc.  
   - The pricer loops over each row, builds the appropriate model (GBM or ML), picks the correct payoff, and prints out the Monte Carlo price & standard error.

## Thought Process and Motivation

- **Traditional Monte Carlo** for option pricing uses GBM. It’s reliable and straightforward, but it may not capture all real-world dynamics.
- **Machine Learning** can supplement or replace GBM by learning patterns from historical data. However, it can produce unstable or extreme outcomes, so we use **noise clipping** and consider log returns to maintain realistic price paths.
- **Exotic options** (barrier, digital, lookback) require specialized payoff functions, but the underlying Monte Carlo engine remains the same.
- **Variance reduction** (antithetic sampling) is a simple technique to improve simulation accuracy with fewer paths.

## How to Run

1. **Clone or download** this repository.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
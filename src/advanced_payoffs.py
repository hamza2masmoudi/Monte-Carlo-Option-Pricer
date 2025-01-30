import numpy as np

def down_and_out_call_payoff(paths, strike, barrier, rebate=0.0):
    """
    Knock-out if price goes below barrier. 
    Else payoff = max(S(T) - strike, 0).
    """
    min_price = np.min(paths, axis=1)
    barrier_hit = (min_price < barrier)

    final_price = paths[:, -1]
    vanilla_call = np.maximum(final_price - strike, 0.0)
    payoff = np.where(barrier_hit, rebate, vanilla_call)
    return payoff

def up_and_out_call_payoff(paths, strike, barrier, rebate=0.0):
    """
    Knock-out if price goes above barrier.
    """
    max_price = np.max(paths, axis=1)
    barrier_hit = (max_price > barrier)

    final_price = paths[:, -1]
    vanilla_call = np.maximum(final_price - strike, 0.0)
    payoff = np.where(barrier_hit, rebate, vanilla_call)
    return payoff

def digital_call_payoff(paths, strike, cash_payout=1.0):
    final_price = paths[:, -1]
    in_the_money = final_price > strike
    return np.where(in_the_money, cash_payout, 0.0)

def digital_put_payoff(paths, strike, cash_payout=1.0):
    final_price = paths[:, -1]
    in_the_money = final_price < strike
    return np.where(in_the_money, cash_payout, 0.0)

def lookback_call_payoff(paths):
    """
    Pays S(T) - min(S(t)) for t in [0, T].
    """
    min_price = np.min(paths, axis=1)
    final_price = paths[:, -1]
    return np.maximum(final_price - min_price, 0.0)

def lookback_put_payoff(paths):
    """
    Pays max(S(t)) - S(T).
    """
    max_price = np.max(paths, axis=1)
    final_price = paths[:, -1]
    return np.maximum(max_price - final_price, 0.0)
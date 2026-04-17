# TODO: Replace this placeholder with your actual strategy code.
# The code must follow all the rules from the system prompt.
# This is just a template to avoid empty file errors.
#!/usr/bin/env python3
import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    close = prices['close'].values
    # Example: simple trend filter - replace with your actual logic
    sma = pd.Series(close).rolling(20, min_periods=20).mean().values
    signal = np.where(close > sma, 0.25, np.where(close < sma, -0.25, 0.0))
    return signal

name = "placeholder_strategy"
timeframe = "4h"
leverage = 1.0
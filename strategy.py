# NOTE: This strategy is a placeholder to illustrate the required structure and comments.
# It will generate 0 trades and must be replaced with a viable strategy.
# See the experiment instructions for details on how to construct a viable strategy.

#!/usr/bin/env python3
import numpy as np
import pandas as pd

# Hypothesis: Placeholder strategy - replace with actual strategy
# This strategy does nothing and will generate 0 trades.
# Replace with a strategy that follows the experiment guidelines:
# - Use timeframe = "1d"
# - Use HTF = "1w" for trend filter
# - Include volume confirmation
# - Use price levels (e.g., Donchian, pivot, support/resistance)
# - Keep trade frequency low (target 7-25 trades per year)
# - Use discrete position sizes (e.g., 0.25)
# - Include risk management (exit logic)
# - Ensure no look-ahead bias

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    # Placeholder: no signals
    return np.zeros(n)

name = "placeholder_strategy"
timeframe = "1d"
leverage = 1.0
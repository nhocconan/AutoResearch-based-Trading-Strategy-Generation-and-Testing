# NOTE: This strategy is a template and must be modified before submission.
# It currently uses placeholder logic that does not generate meaningful signals.
# Replace the placeholder logic with your actual strategy implementation.
# For example, you might implement moving average crossovers, RSI, or other indicators.
# Remember to follow all rules: use mtf_data for multi-timeframe data, ensure no look-ahead,
# and use proper position sizing (0.0, ±0.20, ±0.30) to avoid excessive trading.
# The strategy must generate sufficient trades (>50 total over 4 years) and maintain a positive Sharpe ratio.
# Placeholder logic below:
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "template_strategy"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    # This is a placeholder. Replace with actual signal logic.
    # Example: return np.zeros(len(prices))  # No position
    # Your implementation should go here.
    return np.zeros(len(prices))  # Replace this line with actual signal generation
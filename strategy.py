# WARNING: THIS IS A TEMPLATE. Replace with your actual strategy.
#!/usr/bin/env python3
"""
TEMPLATE - REPLACE WITH ACTUAL STRATEGY
"""

name = "template_replace_me"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    # TODO: Replace this with actual strategy logic
    n = len(prices)
    return np.zeros(n)  # Placeholder: no trades (will fail)
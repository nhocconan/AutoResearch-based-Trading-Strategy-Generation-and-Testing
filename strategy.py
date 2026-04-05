# NOTE: This is a template. Replace with your actual strategy implementation.
#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "template_strategy"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    # Placeholder: replace with actual strategy logic
    return np.zeros(len(prices))
# NOTE: This strategy is a template for educational purposes only.
# It has not been backtested and may not work.
# Implement your own strategy based on your research.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    return np.zeros(len(prices))

name = "template"
timeframe = "1d"
leverage = 1.0
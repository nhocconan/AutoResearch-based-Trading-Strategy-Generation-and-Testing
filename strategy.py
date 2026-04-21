# WARNING: This is a template - DO NOT SUBMIT
# Your strategy MUST generate at least 10 trades per symbol during train (2021-2024)
# and at least 3 trades per symbol during test (2025-2026)
# Strategies with insufficient trades will be auto-rejected
# Focus on creating a robust logic that generates sufficient trading frequency
# while maintaining statistical significance in both bull and bear markets

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Simple moving average crossover system
    close = prices['close'].values
    
    # Fast and slow SMAs
    sma_fast = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    sma_slow = pd.Series(close).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        if np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]):
            continue
            
        # Golden cross: fast MA crosses above slow MA
        if sma_fast[i] > sma_slow[i] and sma_fast[i-1] <= sma_slow[i-1]:
            signals[i] = 0.25
        # Death cross: fast MA crosses below slow MA
        elif sma_fast[i] < sma_slow[i] and sma_fast[i-1] >= sma_slow[i-1]:
            signals[i] = -0.25
        else:
            # Hold current position
            signals[i] = signals[i-1] if i > 0 else 0
    
    return signals

name = "SMA_Crossover_10_30"
timeframe = "4h"
leverage = 1.0
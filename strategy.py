#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with Volume Spike Filter
# Uses Williams %R (14) on 6h for overbought/oversold conditions combined with 
# volume spikes (>2x median) for reversal confirmation. Trades against extreme 
# momentum in ranging markets. Works in both bull (sell rallies) and bear (buy dumps).
# Target: 60-120 total trades over 4 years (15-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume spike filter: > 2x median of last 20 periods
    volume_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(14, n):
        # Skip if Williams %R not ready
        if np.isnan(williams_r[i]) or np.isnan(volume_median[i]):
            continue
            
        # Long: oversold (< -80) + volume spike
        if williams_r[i] < -80 and volume[i] > 2.0 * volume_median[i] and position <= 0:
            position = 1
            signals[i] = base_size
            
        # Short: overbought (> -20) + volume spike
        elif williams_r[i] > -20 and volume[i] > 2.0 * volume_median[i] and position >= 0:
            position = -1
            signals[i] = -base_size
            
        # Exit: Williams %R returns to neutral range (-50 to -50) or opposite extreme
        elif position == 1 and (williams_r[i] > -50 or williams_r[i] < -95):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r[i] < -50 or williams_r[i] > -5):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Volume_Spike_MeanReversion"
timeframe = "6h"
leverage = 1.0
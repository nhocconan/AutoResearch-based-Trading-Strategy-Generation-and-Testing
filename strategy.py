# 12h_OvernightBreakout_With_Volume_Spike
# Hypothesis: Overnight price moves (21:00-09:00 UTC) often break the prior session's range.
# In BTC/ETH, these moves are driven by Asian session momentum and can be filtered by volume spikes.
# Breakouts above/below the prior session high/low are taken only with volume > 2x median.
# Works in both bull and bear markets as it captures momentum bursts regardless of direction.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for session range calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Previous session high/low (shifted by 1 bar)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    
    # Align previous session range to current timeframe
    prev_high_12h_aligned = align_htf_to_ltf(prices, df_12h, prev_high_12h)
    prev_low_12h_aligned = align_htf_to_ltf(prices, df_12h, prev_low_12h)
    
    # Volume spike filter: current volume > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):
        # Skip if required data is NaN
        if (np.isnan(prev_high_12h_aligned[i]) or 
            np.isnan(prev_low_12h_aligned[i]) or
            np.isnan(vol_median[i])):
            continue
        
        # Long entry: break above prior session high + volume spike
        if (close[i] > prev_high_12h_aligned[i] and
            volume[i] > 2.0 * vol_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: break below prior session low + volume spike
        elif (close[i] < prev_low_12h_aligned[i] and
              volume[i] > 2.0 * vol_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout (opposite session level)
        elif position == 1 and close[i] < prev_low_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > prev_high_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_OvernightBreakout_With_Volume_Spike"
timeframe = "12h"
leverage = 1.0
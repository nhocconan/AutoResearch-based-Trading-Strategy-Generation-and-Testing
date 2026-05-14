#!/usr/bin/env python3
# 6h_1d_volume_breakout_v1
# Hypothesis: 6-hour breakouts above/below daily price high/low with volume confirmation, exiting at opposite daily extreme.
# Uses volume > 2x 20-period average to filter breakouts, reducing false signals.
# Works in bull markets by catching upward breakouts, in bear markets by catching downward breakdowns.
# Target: 15-40 trades per year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for daily high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily high and low
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Align 1d levels to 6h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below daily low
            if close[i] <= daily_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above daily high
            if close[i] >= daily_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above daily high with volume confirmation
            if close[i] > daily_high_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below daily low with volume confirmation
            elif close[i] < daily_low_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals
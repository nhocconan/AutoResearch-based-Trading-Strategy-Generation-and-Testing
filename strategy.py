#!/usr/bin/env python3
# 1h_4d_camarilla_breakout_v1
# Hypothesis: 1-hour breakouts above/below 4-day Camarilla pivot levels (H4/L4) with volume confirmation.
# Long when price breaks above H4 with volume confirmation.
# Short when price breaks below L4 with volume confirmation.
# Exit when price returns to the 4-day pivot point (PP).
# Uses 4-day timeframe for signal direction, 1h for entry timing.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Session filter: 08-20 UTC to reduce noise trades.
# Position size: 0.20.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4d data ONCE before loop for Camarilla pivot levels
    df_4d = get_htf_data(prices, '4d')
    if len(df_4d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 4d bar
    high_4d = df_4d['high'].values
    low_4d = df_4d['low'].values
    close_4d = df_4d['close'].values
    
    # Camarilla formulas
    pp_4d = (high_4d + low_4d + close_4d) / 3.0
    range_4d = high_4d - low_4d
    
    # H4 and L4 levels (stronger breakout levels)
    h4_4d = close_4d + (range_4d * 1.1 / 2)
    l4_4d = close_4d - (range_4d * 1.1 / 2)
    
    # Align 4d levels to 1h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_4d, pp_4d)
    h4_aligned = align_htf_to_ltf(prices, df_4d, h4_4d)
    l4_aligned = align_htf_to_ltf(prices, df_4d, l4_4d)
    
    # Volume confirmation - 24 period average (24h)
    vol_ma_24 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # pre-compute before loop
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma_24[i]) or
            not (8 <= hours[i] <= 20)):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below Pivot Point
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Pivot Point
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price breaks above H4 level with volume confirmation
            if close[i] > h4_aligned[i] and volume[i] > vol_ma_24[i] * 1.5:
                position = 1
                signals[i] = 0.20
            # Enter short: price breaks below L4 level with volume confirmation
            elif close[i] < l4_aligned[i] and volume[i] > vol_ma_24[i] * 1.5:
                position = -1
                signals[i] = -0.20
    
    return signals
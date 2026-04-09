#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v2
# Hypothesis: 12-hour breakouts above/below daily Camarilla pivot levels (H4/L4) with volume confirmation and tighter entry conditions.
# Uses breakout of H4/L4 levels for higher probability moves. Exit when price returns to the daily pivot point (PP).
# Works in both bull and bear markets as pivot levels adapt to volatility.
# Added volume filter (volume > 2x average) and stricter breakout criteria to reduce trade frequency.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # H4 and L4 levels (stronger breakout levels)
    h4_1d = close_1d + (range_1d * 1.1 / 2)  # Same as R4
    l4_1d = close_1d - (range_1d * 1.1 / 2)  # Same as S4
    
    # Align 1d levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume confirmation - 50 period average with stricter threshold
    vol_ma_50 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 50:
            vol_sum -= volume[i-50]
        if i >= 49:
            vol_ma_50[i] = vol_sum / 50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(pp_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(vol_ma_50[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below Pivot Point
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Pivot Point
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H4 level with volume confirmation (volume > 2x average)
            if close[i] > h4_aligned[i] and volume[i] > vol_ma_50[i] * 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L4 level with volume confirmation (volume > 2x average)
            elif close[i] < l4_aligned[i] and volume[i] > vol_ma_50[i] * 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals
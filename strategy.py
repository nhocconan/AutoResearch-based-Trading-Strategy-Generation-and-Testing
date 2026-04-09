#!/usr/bin/env python3
# 1h_1d_camarilla_breakout_v1
# Hypothesis: 1-hour breakouts above/below daily Camarilla pivot levels (H4/L4) with volume confirmation and exit at pivot point.
# Uses tighter entry conditions (volume > 2.5x 20-period average) to reduce trades and avoid fee drag.
# Works in bull markets by catching breakouts, in bear markets by fading false breaks via pivot reversion.
# Target: 15-37 trades per year per symbol (60-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_1d_camarilla_breakout_v1"
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
    h4_1d = close_1d + (range_1d * 1.1 / 2)
    l4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align 1d levels to 1h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Session filter: 8-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(pp_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        if not (8 <= hours[i] <= 20):
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
            if close[i] > h4_aligned[i] and volume[i] > vol_ma_20[i] * 2.5:
                position = 1
                signals[i] = 0.20
            # Enter short: price breaks below L4 level with volume confirmation
            elif close[i] < l4_aligned[i] and volume[i] > vol_ma_20[i] * 2.5:
                position = -1
                signals[i] = -0.20
    
    return signals
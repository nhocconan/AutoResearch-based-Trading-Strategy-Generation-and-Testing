#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v1
# Hypothesis: Daily breakouts above/below weekly Camarilla pivot levels (H4/L4) with volume confirmation.
# Weekly pivot levels act as strong support/resistance that adapt to volatility.
# Breaking above weekly H4 indicates bullish momentum; breaking below L4 indicates bearish momentum.
# Works in both bull and bear markets as weekly pivot levels adapt to volatility, and volume filter reduces whipsaw.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla formulas
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # H4 and L4 levels (stronger breakout levels)
    h4_1w = close_1w + (range_1w * 1.1 / 2)  # Same as R4
    l4_1w = close_1w - (range_1w * 1.1 / 2)  # Same as S4
    
    # Align 1w levels to 1d timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4_1w)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4_1w)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_spike = volume > vol_ma_20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(pp_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below Weekly Pivot Point
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Weekly Pivot Point
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly H4 level with volume confirmation
            if close[i] > h4_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly L4 level with volume confirmation
            elif close[i] < l4_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals
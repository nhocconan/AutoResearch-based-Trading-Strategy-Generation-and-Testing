#!/usr/bin/env python3
# 6h_1w_1d_camarilla_breakout_v1
# Hypothesis: 6-hour breakouts above/below daily Camarilla pivot levels (H4/L4) with weekly trend filter and volume confirmation.
# Long when weekly trend is up (price > weekly SMA50) and price breaks H4 with volume confirmation.
# Short when weekly trend is down (price < weekly SMA50) and price breaks L4 with volume confirmation.
# Exit when price returns to the daily pivot point (PP).
# Weekly trend filter ensures we trade with the higher timeframe momentum, reducing whipsaws in choppy markets.
# Works in both bull and bear markets as pivot levels adapt to volatility and weekly filter adapts to trend.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_camarilla_breakout_v1"
timeframe = "6h"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
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
    
    # Calculate weekly SMA50 for trend filter
    close_1w = df_1w['close'].values
    sma50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        sma_sum = 0
        for i in range(len(close_1w)):
            sma_sum += close_1w[i]
            if i >= 50:
                sma_sum -= close_1w[i-50]
            if i >= 49:
                sma50_1w[i] = sma_sum / 50
    
    # Align 1d levels and weekly trend to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
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
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(pp_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(sma50_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
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
            # Enter long: weekly trend up (price > weekly SMA50) AND price breaks above H4 level with volume confirmation
            if close[i] > sma50_1w_aligned[i] and close[i] > h4_aligned[i] and volume[i] > vol_ma_20[i] * 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: weekly trend down (price < weekly SMA50) AND price breaks below L4 level with volume confirmation
            elif close[i] < sma50_1w_aligned[i] and close[i] < l4_aligned[i] and volume[i] > vol_ma_20[i] * 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals
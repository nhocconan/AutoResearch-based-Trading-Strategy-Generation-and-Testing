#!/usr/bin/env python3
"""
1d_1w_Pivot_Breakout_Trend_Volume_v5
Hypothesis: Daily breakouts at previous week's high/low with weekly EMA trend filter and volume confirmation.
Trades in direction of weekly trend using weekly pivot levels for precise entry. Works in both bull and bear
markets by aligning with higher timeframe trend. Volume confirmation filters false breakouts. Low trade frequency
reduces fee drag.
"""

name = "1d_1w_Pivot_Breakout_Trend_Volume_v5"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 12:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema12_1w = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_1w_aligned = align_htf_to_ltf(prices, df_1w, ema12_1w)
    
    # === Weekly Data for Pivot Levels (previous week's high/low) ===
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC
    ph_1w = df_1w['high'].values
    pl_1w = df_1w['low'].values
    
    # Pivot levels: using previous week's high and low for breakout
    pivot_high = ph_1w  # Previous week's high
    pivot_low = pl_1w   # Previous week's low
    
    # Align to daily timeframe
    pivot_high_1d = align_htf_to_ltf(prices, df_1w, pivot_high)
    pivot_low_1d = align_htf_to_ltf(prices, df_1w, pivot_low)
    
    # === Volume Filter (1.5x 20-period EMA on daily) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly EMA and weekly data)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_high_1d[i]) or np.isnan(pivot_low_1d[i]) or 
            np.isnan(ema12_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above previous week's high with uptrend and volume
            if (close[i] > pivot_high_1d[i] and 
                close[i] > ema12_1w_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below previous week's low with downtrend and volume
            elif (close[i] < pivot_low_1d[i] and 
                  close[i] < ema12_1w_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below previous week's low (mean reversion)
            if close[i] < pivot_low_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above previous week's high (mean reversion)
            if close[i] > pivot_high_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
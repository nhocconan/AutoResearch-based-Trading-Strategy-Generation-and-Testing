#!/usr/bin/env python3
"""
4h_HighLowBreakout_Trend_Volume_v1
Hypothesis: Breakouts at daily high/low levels with 12h EMA trend filter and volume confirmation.
This strategy trades breakouts in the direction of the 12h trend, using the previous day's high/low
as breakout levels. Works in both bull and bear markets by aligning with higher timeframe trend.
Volume confirmation filters out false breakouts. Low trade frequency reduces fee drag.
"""

name = "4h_HighLowBreakout_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Data for Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 12:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema12 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_aligned = align_htf_to_ltf(prices, df_12h, ema12)
    
    # === Daily Data for Breakout Levels (previous day's high/low) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    ph_1d = df_1d['high'].values
    pl_1d = df_1d['low'].values
    
    # Breakout levels: previous day's high and low
    breakout_high = ph_1d  # Previous day's high
    breakout_low = pl_1d   # Previous day's low
    
    # Align to 4h timeframe
    breakout_high_4h = align_htf_to_ltf(prices, df_1d, breakout_high)
    breakout_low_4h = align_htf_to_ltf(prices, df_1d, breakout_low)
    
    # === Volume Filter (1.5x 20-period EMA on 4h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 12h EMA and daily data)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(breakout_high_4h[i]) or np.isnan(breakout_low_4h[i]) or 
            np.isnan(ema12_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above previous day's high with uptrend and volume
            if (close[i] > breakout_high_4h[i] and 
                close[i] > ema12_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.30
                position = 1
            # Short breakdown: price closes below previous day's low with downtrend and volume
            elif (close[i] < breakout_low_4h[i] and 
                  close[i] < ema12_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price closes below previous day's low (mean reversion)
            if close[i] < breakout_low_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price closes above previous day's high (mean reversion)
            if close[i] > breakout_high_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals
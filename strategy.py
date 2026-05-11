#!/usr/bin/env python3
"""
1d_RVOL_Breakout_v1
Hypothesis: On the daily timeframe, breakouts from the previous day's high/low
with above-average volume (RVOL > 1.5) capture momentum moves that persist
through the next day. Works in bull markets (breakouts continue up) and bear
markets (breakdowns continue down) by following the direction of the breakout.
Uses 1-week EMA as a trend filter to avoid counter-trend trades. Target: 30-100
trades over 4 years (7-25/year) on 1d timeframe.
"""

name = "1d_RVOL_Breakout_v1"
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
    
    # === 1D Data for Reference Levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # Previous day's high/low (reference for breakout)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_vol_1d = np.roll(vol_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_vol_1d[0] = np.nan
    
    # Align to 1d (no shift needed as we use previous day's values)
    prev_high_aligned = prev_high_1d
    prev_low_aligned = prev_low_1d
    prev_vol_aligned = prev_vol_1d
    
    # === 1W Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1-week EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === RVOL Calculation (Relative Volume) ===
    # 20-day average volume for RVOL denominator
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    rvol = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # need 20 for vol MA + buffers
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or 
            np.isnan(prev_vol_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(rvol[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: close > prev day high + RVOL > 1.5 + above weekly EMA
            if (close[i] > prev_high_aligned[i] and 
                rvol[i] > 1.5 and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: close < prev day low + RVOL > 1.5 + below weekly EMA
            elif (close[i] < prev_low_aligned[i] and 
                  rvol[i] > 1.5 and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below previous day's low (failed breakout) or RVOL drops
            if close[i] < prev_low_aligned[i] or rvol[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: close above previous day's high (failed breakdown) or RVOL drops
            if close[i] > prev_high_aligned[i] or rvol[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
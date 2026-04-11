#!/usr/bin/env python3
# 4h_1d_rvol_breakout_v1
# Strategy: 4h RVOL-based breakout with 1-day trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Combines relative volume (RVOL) spikes with price breaking 20-period high/low
# and 1-day EMA50 trend filter to capture momentum bursts in both bull and bear markets.
# RVOL > 2.0 indicates institutional interest; breakout confirms direction; EMA50 filter
# avoids counter-trend trades. Designed for low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rvol_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period high/low for breakout detection
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_20 = high_series.rolling(window=20, min_periods=20).max().values
    lowest_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average for RVOL calculation
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # RVOL: current volume / 20-period average volume
        rvol = volume[i] / vol_avg_20[i] if vol_avg_20[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = high[i] > highest_20[i-1]  # Current high exceeds prior 20-period high
        breakout_down = low[i] < lowest_20[i-1]  # Current low exceeds prior 20-period low
        
        # Entry logic: RVOL spike + breakout + trend alignment
        if (rvol > 2.0 and breakout_up and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (rvol > 2.0 and breakout_down and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or RVOL normalization
        elif position == 1 and (not uptrend or rvol < 1.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or rvol < 1.2):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
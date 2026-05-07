#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend"
timeframe = "12h"
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
    
    # 1-day trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # 12-hour Camarilla levels (using previous day's range)
    # Calculate pivot and levels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d_vals) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla R1 and S1
    R1 = close_1d_vals + (range_1d * 1.1 / 12)
    S1 = close_1d_vals - (range_1d * 1.1 / 12)
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation (20-period average)
    vol_ma = np.zeros(n)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA and Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and 1-day uptrend
            if close[i] > R1_aligned[i] and vol_surge and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge and 1-day downtrend
            elif close[i] < S1_aligned[i] and vol_surge and trend_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below pivot OR trend turns down
            if close[i] < pivot_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above pivot OR trend turns up
            if close[i] > pivot_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakouts with volume surge and 1-day trend filter
# Captures institutional breakout patterns in both bull and bear markets.
# Long when price breaks above R1 with volume surge in 1-day uptrend.
# Short when price breaks below S1 with volume surge in 1-day downtrend.
# Uses 12h timeframe for entries/exits with 1-day trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakout validity. Position size 0.25 manages risk.
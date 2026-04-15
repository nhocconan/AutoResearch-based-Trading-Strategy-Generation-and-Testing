#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12h-1d Dual Timeframe Confluence Strategy
# Uses 12h EMA50 as trend filter and 1d high/low as key support/resistance levels.
# Long when: price > 12h EMA50 AND breaks above 1d high with volume confirmation
# Short when: price < 12h EMA50 AND breaks below 1d low with volume confirmation
# Works in bull markets (buy dips above EMA50) and bear markets (sell rallies below EMA50).
# Target: 80-150 total trades over 4 years (20-38/year).
# Timeframe: 4h, HTF: 12h/1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Load 1d data for daily high/low levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Previous day's high and low (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Align previous day's high/low to 4h timeframe
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(prev_high_1d_aligned[i]) or 
            np.isnan(prev_low_1d_aligned[i])):
            continue
        
        # Volume confirmation: current volume > 1.5x median of last 20 bars
        vol_median = np.median(volume[max(0, i-20):i+1])
        vol_ok = volume[i] > 1.5 * vol_median
        
        # Long entry: price above 12h EMA50 AND breaks above 1d high + volume
        if (close[i] > ema50_12h_aligned[i] and
            close[i] > prev_high_1d_aligned[i] and
            vol_ok and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below 12h EMA50 AND breaks below 1d low + volume
        elif (close[i] < ema50_12h_aligned[i] and
              close[i] < prev_low_1d_aligned[i] and
              vol_ok and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout (price crosses back below/above EMA50)
        elif position == 1 and close[i] < ema50_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema50_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_12hEMA50_1dLevel_Breakout"
timeframe = "4h"
leverage = 1.0
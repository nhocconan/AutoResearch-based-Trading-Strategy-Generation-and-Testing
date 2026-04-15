#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Williams %R + 1d volume + 12h trend filter
# Williams %R identifies overbought/oversold conditions; volume confirms momentum; 12h EMA50 ensures trend alignment.
# Works in both bull and bear markets by fading extremes in range and following momentum in trends.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1d data for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R (14-period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 12h for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            continue
        
        # Long entry: Williams %R oversold (< -80) + volume spike + price above 12h EMA50
        if (williams_r_aligned[i] < -80 and 
            volume[i] > 1.5 * vol_avg_aligned[i] and 
            close[i] > ema50_12h_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R overbought (> -20) + volume spike + price below 12h EMA50
        elif (williams_r_aligned[i] > -20 and 
              volume[i] > 1.5 * vol_avg_aligned[i] and 
              close[i] < ema50_12h_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or Williams %R returns to neutral range (-50 to -50)
        elif position == 1 and williams_r_aligned[i] > -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and williams_r_aligned[i] < -50:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_1dVolume_12hEMA_Trend"
timeframe = "6h"
leverage = 1.0
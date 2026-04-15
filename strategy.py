#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) + 1d Williams %R (14) + volume confirmation
# Uses short-term oversold/overbought conditions on 6h confirmed by longer-term momentum on 1d.
# In bull markets: buy 6m %R < -80 when 1d %R > -50 (bullish momentum).
# In bear markets: sell 6m %R > -20 when 1d %R < -50 (bearish momentum).
# Volume filter ensures momentum is real. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Williams %R
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1d data for longer-term Williams %R and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R (14-period) on 6h
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_6h = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    wr_6h = (highest_high_6h - close_6h) / (highest_high_6h - lowest_low_6h + 1e-10) * -100
    
    # Calculate Williams %R (14-period) on 1d
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    wr_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d + 1e-10) * -100
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    wr_6h_aligned = align_htf_to_ltf(prices, df_6h, wr_6h)
    wr_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(wr_6h_aligned[i]) or np.isnan(wr_1d_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: 6m %R oversold (< -80) + 1d %R bullish (> -50) + volume confirmation
        if (wr_6h_aligned[i] < -80 and
            wr_1d_aligned[i] > -50 and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: 6m %R overbought (> -20) + 1d %R bearish (< -50) + volume confirmation
        elif (wr_6h_aligned[i] > -20 and
              wr_1d_aligned[i] < -50 and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite signal or momentum deterioration
        elif position == 1 and (wr_6h_aligned[i] > -20 or wr_1d_aligned[i] < -50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (wr_6h_aligned[i] < -80 or wr_1d_aligned[i] > -50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Momentum_Confirm"
timeframe = "6h"
leverage = 1.0
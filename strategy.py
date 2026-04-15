#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-day trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In trending markets (ADX > 25 on 1d),
# we fade extreme readings: short when %R > -20 (overbought), long when %R < -80 (oversold).
# In ranging markets (ADX <= 25), we mean-revert at extreme levels.
# Volume confirmation ensures breakouts have conviction.
# Works in bull markets (fade bounces in downtrends) and bear markets (fade rallies in uptrends).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # Calculate ADX (14-period) on 1d for trend filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align Williams %R and ADX to 6h timeframe
    willr_aligned = align_htf_to_ltf(prices, df_1d, willr)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(willr_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: Williams %R oversold (< -80) + volume confirmation
        if (willr_aligned[i] < -80 and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R overbought (> -20) + volume confirmation
        elif (willr_aligned[i] > -20 and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Williams %R crosses back through -50 (mean reversion) or ADX < 20 (ranging)
        elif position == 1 and (willr_aligned[i] > -50 or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (willr_aligned[i] < -50 or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Volume_ADX_Filter"
timeframe = "6h"
leverage = 1.0
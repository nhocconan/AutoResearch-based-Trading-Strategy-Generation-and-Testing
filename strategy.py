# 6h_DailyPivot_Breakout_Volume_TrendFilter
# Breakout of previous day's high/low with volume confirmation and trend filter.
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Uses 1d pivot levels + volume spike + 1d ADX trend filter.
# Target: 50-150 total trades over 4 years = 12-37/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for previous day's high/low and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's high and low (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan  # First value has no previous day
    prev_low_1d[0] = np.nan
    
    # Align previous day's high/low to 6h timeframe
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
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
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high_1d_aligned[i]) or np.isnan(prev_low_1d_aligned[i]) or
            np.isnan(adx_aligned[i])):
            continue
        
        # Volume spike: current volume > 1.5x median of last 20 bars
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_spike = volume[i] > 1.5 * vol_median
        
        # Long entry: price breaks above previous day's high + volume spike + ADX > 25
        if (close[i] > prev_high_1d_aligned[i] and
            volume_spike and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below previous day's low + volume spike + ADX > 25
        elif (close[i] < prev_low_1d_aligned[i] and
              volume_spike and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < prev_low_1d_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prev_high_1d_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_DailyPivot_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0
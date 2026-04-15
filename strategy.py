#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ADX trend filter
# Uses 20-period Donchian channels (highest high/lowest low over 20 periods) for breakouts.
# Long when price breaks above upper band with volume > 1.5x median and ADX > 25.
# Short when price breaks below lower band with volume > 1.5x median and ADX > 25.
# Exits on reverse breakout or ADX < 20 (ranging market).
# Designed to work in both bull and bear markets by capturing breakouts in direction of trend.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian channels (to avoid look-ahead and use actual 4h bars)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Load 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 15m timeframe (since we're using 4h for calculation)
    # Note: For 4h primary timeframe, we need to align 4h Donchian to 4h bars
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low)
    
    # Calculate ADX (14-period) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
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
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or
            np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian upper band + volume confirmation + ADX > 25
        if (close[i] > highest_high_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian lower band + volume confirmation + ADX > 25
        elif (close[i] < lowest_low_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < lowest_low_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_high_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0
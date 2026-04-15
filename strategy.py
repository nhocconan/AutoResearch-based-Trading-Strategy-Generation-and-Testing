#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Bollinger Breakout with Volume Confirmation and ADX Trend Filter
# Uses the previous week's Bollinger Bands (20, 2) as support/resistance levels. 
# Breakouts above upper band or below lower band are traded only when confirmed by volume and ADX > 25 (trending market).
# Works in bull markets (breakouts up) and bear markets (breakouts down). Target: 30-100 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20, 2) on weekly
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Previous week's bands (shifted by 1 to avoid look-ahead)
    prev_upper = np.roll(upper_band, 1)
    prev_lower = np.roll(lower_band, 1)
    prev_upper[0] = np.nan  # First value has no previous week
    prev_lower[0] = np.nan
    
    # Align previous week's bands to daily timeframe
    prev_upper_aligned = align_htf_to_ltf(prices, df_1w, prev_upper)
    prev_lower_aligned = align_htf_to_ltf(prices, df_1w, prev_lower)
    
    # Load 1d data for ADX trend filter (using daily data itself)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
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
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_upper_aligned[i]) or np.isnan(prev_lower_aligned[i]) or
            np.isnan(adx[i])):
            continue
        
        # Long entry: price breaks above previous week's upper band + volume confirmation + ADX > 25
        if (close[i] > prev_upper_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below previous week's lower band + volume confirmation + ADX > 25
        elif (close[i] < prev_lower_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < prev_lower_aligned[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prev_upper_aligned[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Weekly_Bollinger_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0
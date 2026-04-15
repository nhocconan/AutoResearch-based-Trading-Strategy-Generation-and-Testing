#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with Volume and ADX Filter
# Uses daily Camarilla pivot levels (calculated from previous day's range) as support/resistance.
# Long when price breaks above H3 level with volume > 1.5x median and ADX > 25.
# Short when price breaks below L3 level with volume > 1.5x median and ADX > 25.
# Exits when price returns to Pivot point or ADX drops below 20 (ranging market).
# Works in trending markets (breakouts) and avoids ranging conditions via ADX filter.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day using previous day's range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.5 * (high - low)
    # H1 = close + 0.25 * (high - low)
    # Pivot = (high + low + close) / 3
    # L1 = close - 0.25 * (high - low)
    # L2 = close - 0.5 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    
    # First day has no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    H3 = prev_close + 1.0 * range_val
    L3 = prev_close - 1.0 * range_val
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate ADX (14-period) on 12h for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
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
    
    # Smoothed values with proper min_periods
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: price breaks above H3 + volume confirmation + ADX > 25
        if (close[i] > H3_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below L3 + volume confirmation + ADX > 25
        elif (close[i] < L3_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price returns to pivot level or ADX drops below 20 (ranging market)
        elif position == 1 and (close[i] < pivot_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > pivot_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0
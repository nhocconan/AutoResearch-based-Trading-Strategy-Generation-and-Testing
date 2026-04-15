#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot + Volume + ADX Trend Filter
# Uses weekly pivot points (R1, S1) from 1w data as support/resistance.
# Long: price crosses above R1 with volume spike and ADX > 25 (trending)
# Short: price crosses below S1 with volume spike and ADX > 25
# Works in bull/bear: buys strength, sells weakness. Target: 50-150 total trades.
# Timeframe: 6h, HTF: 1w

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Shift by 1 to avoid look-ahead (use previous week's levels)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_prev)
    
    # Load 12h data for ADX trend filter (more stable than 6h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(close_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(close_12h, 1)), 
                        np.maximum(np.roll(close_12h, 1) - low_12h, 0), 0)
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
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(adx_aligned[i])):
            continue
        
        # Volume condition: current volume > 1.5 * median of last 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        vol_ok = volume[i] > 1.5 * vol_median
        
        # Long entry: price crosses above R1 + volume + ADX > 25
        if (close[i] > r1_aligned[i] and
            vol_ok and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price crosses below S1 + volume + ADX > 25
        elif (close[i] < s1_aligned[i] and
              vol_ok and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse pivot level or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < s1_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > r1_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Volume_ADX"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Daily Pivot + Volume + ADX Trend Filter
# Uses daily pivot points (support/resistance) as key levels. Long when price > R1 with volume confirmation and ADX > 25 (trending).
# Short when price < S1 with volume confirmation and ADX > 25. Works in bull/bear markets by trading with the trend.
# Target: 50-150 total trades over 4 years. Timeframe: 4h, HTF: 1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Previous day's pivot levels (shifted by 1 to avoid look-ahead)
    pivot_prev = np.roll(pivot, 1)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    pivot_prev[0] = np.nan
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    # Align to 4h timeframe
    pivot_prev_aligned = align_htf_to_ltf(prices, df_1d, pivot_prev)
    r1_prev_aligned = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_prev_aligned = align_htf_to_ltf(prices, df_1d, s1_prev)
    
    # Load 4h data for ADX trend filter (using same timeframe)
    high_4h = high
    low_4h = low
    close_4h = close
    
    # Calculate ADX (14-period) on 4h
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(close_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(close_4h, 1)), 
                        np.maximum(np.roll(close_4h, 1) - low_4h, 0), 0)
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
        if (np.isnan(r1_prev_aligned[i]) or np.isnan(s1_prev_aligned[i]) or
            np.isnan(adx[i])):
            continue
        
        # Long entry: price above R1 + volume confirmation + ADX > 25
        if (close[i] > r1_prev_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below S1 + volume confirmation + ADX > 25
        elif (close[i] < s1_prev_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse to opposite level or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < s1_prev_aligned[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > r1_prev_aligned[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_DailyPivot_Volume_ADX"
timeframe = "4h"
leverage = 1.0
#%%
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1-week pivot resistance/support and volume confirmation.
In strong trends, price tends to break through weekly pivot levels (R4/S4) with volume expansion.
Enters long on breakout above weekly R4 with volume > 1.5x average, short on breakdown below weekly S4.
Uses 1-day ADX > 25 to confirm trending regime and avoid false breakouts in ranging markets.
Exits when price returns to weekly pivot point or ADX drops below 20.
Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.
"""

name = "6h_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate weekly pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly OHLC
    weekly_high = df_1w['high']
    weekly_low = df_1w['low']
    weekly_close = df_1w['close']
    weekly_open = df_1w['open']
    
    # Pivot point calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support and resistance levels
    s1 = (2 * pivot) - weekly_high
    r1 = (2 * pivot) - weekly_low
    s2 = pivot - (weekly_high - weekly_low)
    r2 = pivot + (weekly_high - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - weekly_close)
    r4 = weekly_high + 3 * (weekly_close - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_vals = pivot.values
    r4_vals = r4.values
    s4_vals = s4.values
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_vals)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_vals)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_vals)
    
    # 1-day ADX for trend strength (avoid ranging markets)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d.shift(1))
    tr3 = np.abs(low_1d - close_1d.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - high_1d.shift(1)) > (low_1d.shift(1) - low_1d), 
                       np.maximum(high_1d - high_1d.shift(1), 0), 0)
    dm_minus = np.where((low_1d.shift(1) - low_1d) > (high_1d - high_1d.shift(1)), 
                        np.maximum(low_1d.shift(1) - low_1d, 0), 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_14 / tr_14)
    di_minus = 100 * (dm_minus_14 / tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_ratio = volume / volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > weekly R4 + ADX > 25 (trending) + volume > 1.5x average
            if (close[i] > r4_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price < weekly S4 + ADX > 25 (trending) + volume > 1.5x average
            elif (close[i] < s4_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly pivot OR ADX drops below 20 (trend weakening)
            if (close[i] <= pivot_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot OR ADX drops below 20 (trend weakening)
            if (close[i] >= pivot_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#%%
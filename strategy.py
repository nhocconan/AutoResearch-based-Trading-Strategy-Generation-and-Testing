#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Supertrend_MultiTF_Trend_With_Weekly_Pivot_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (based on previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly support/resistance levels
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + range_1w
    s2_1w = pivot_1w - range_1w
    
    # Get daily data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for Supertrend (7-period ATR, multiplier 3.0)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/7, adjust=False, min_periods=7).mean().values
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1d, np.nan, dtype=float)
    dir_ = np.full_like(close_1d, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > upper_band[i-1]:
            dir_[i] = 1
        elif close_1d[i] < lower_band[i-1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i-1]
            if dir_[i] == -1 and lower_band[i] > lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if dir_[i] == 1 and upper_band[i] < upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if dir_[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align all indicators to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    dir_aligned = align_htf_to_ltf(prices, df_1d, dir_.astype(float))
    
    # Volume filters
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(dir_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]  # Volume filter
        
        if position == 0:
            # Long: Weekly pivot support + daily Supertrend uptrend + volume
            if (close[i] > s1_1w_aligned[i] and  # Above weekly S1
                dir_aligned[i] == 1 and         # Daily Supertrend uptrend
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: Weekly pivot resistance + daily Supertrend downtrend + volume
            elif (close[i] < r1_1w_aligned[i] and  # Below weekly R1
                  dir_aligned[i] == -1 and         # Daily Supertrend downtrend
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below weekly S1 or Supertrend turns down
            if (close[i] < s1_1w_aligned[i] or 
                dir_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above weekly R1 or Supertrend turns up
            if (close[i] > r1_1w_aligned[i] or 
                dir_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
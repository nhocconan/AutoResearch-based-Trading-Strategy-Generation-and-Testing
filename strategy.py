#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_Volume_Filter
Strategy: Weekly Pivot R1/S1 breakout with daily volume confirmation and EMA trend filter.
Long: Price breaks above weekly R1 pivot with volume > 1.5x 20-day average and price > daily EMA50.
Short: Price breaks below weekly S1 pivot with volume > 1.5x 20-day average and price < daily EMA50.
Exit: Price crosses back below/above the pivot level or trend changes.
Designed for 1d timeframe: ~10-20 trades/year per symbol (40-80 total over 4 years).
Works in bull/bear via EMA trend filter and pivot level structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Points (standard formula)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Pivot breakout conditions
        break_above_r1 = high[i] > r1_aligned[i] and close[i] > r1_aligned[i]
        break_below_s1 = low[i] < s1_aligned[i] and close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + break above R1
            if uptrend and vol_confirm and break_above_r1:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + break below S1
            elif downtrend and vol_confirm and break_below_s1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or price back below pivot
            if not uptrend or vol_confirm or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or price back above pivot
            if not downtrend or vol_confirm or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0
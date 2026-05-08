#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_Range_Scalper_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points (Camarilla)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Handle first day
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # Pivot point and range
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4)
    r2 = pivot + (range_1d * 1.1 / 6)
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    s2 = pivot - (range_1d * 1.1 / 6)
    s3 = pivot - (range_1d * 1.1 / 4)
    
    # Align to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily ATR for regime filter
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr30_1d = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    atr30_1d_aligned = align_htf_to_ltf(prices, df_1d, atr30_1d)
    atr_ratio = atr10_1d_aligned / (atr30_1d_aligned + 1e-10)
    
    # Regime: trending if ATR ratio > 1.1, ranging if < 0.9
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or
            np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(atr_ratio[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime
            is_trending = atr_ratio[i] > 1.1
            is_ranging = atr_ratio[i] < 0.9
            
            if is_trending:
                # Trending: breakout at R3/S3 with volume
                long_cond = (close[i] > r3_12h[i] and volume[i] > vol_ma20[i])
                short_cond = (close[i] < s3_12h[i] and volume[i] > vol_ma20[i])
            elif is_ranging:
                # Ranging: mean reversion at R1/S1
                long_cond = (close[i] < s1_12h[i] and volume[i] > vol_ma20[i])
                short_cond = (close[i] > r1_12h[i] and volume[i] > vol_ma20[i])
            else:
                # Transition zone: no trades
                long_cond = False
                short_cond = False
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit
            if is_trending:
                # In trend, exit on reversal below R2
                exit_cond = close[i] < r2_12h[i]
            else:
                # In range, exit at R1 (profit target)
                exit_cond = close[i] > r1_12h[i]
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit
            if is_trending:
                # In trend, exit on reversal above S2
                exit_cond = close[i] > s2_12h[i]
            else:
                # In range, exit at S1 (profit target)
                exit_cond = close[i] < s1_12h[i]
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla pivot-based range scalper that adapts to market regime.
# In trending markets (rising ATR): breakout trades at extreme levels (R3/S3) with volume confirmation.
# In ranging markets: mean reversion at R1/S1 levels.
# Uses 12h timeframe to target 50-150 trades over 4 years (12-37/year) minimizing fee drag.
# Discrete sizing (0.25) reduces churn. Works in both bull (trend following) and bear (mean reversion in ranges).
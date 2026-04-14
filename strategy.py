#!/usr/bin/env python3
"""
6h_1W_Pivot_Trend_v1
Hypothesis: On 6h timeframe, use weekly pivot points (PP, R1, S1) from the previous week for trend-following breakouts with volume confirmation.
Buy when price breaks above weekly R1 with volume > 1.5x 20-period average.
Sell when price breaks below weekly S1 with volume > 1.5x 20-period average.
Exit when price returns to weekly PP (mean reversion) or opposite breakout occurs.
Uses weekly timeframe for structure, 6f for execution - avoids whipsaws in ranging markets while capturing trends.
Works in bull markets (breakouts continuation) and bear markets (breakdown continuations).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pivot_w = np.full_like(close_w, np.nan)
    r1_w = np.full_like(close_w, np.nan)
    s1_w = np.full_like(close_w, np.nan)
    
    for i in range(1, len(close_w)):  # Start from 1 to use previous week
        if np.isnan(high_w[i-1]) or np.isnan(low_w[i-1]) or np.isnan(close_w[i-1]):
            continue
        phigh = high_w[i-1]
        plow = low_w[i-1]
        pclose = close_w[i-1]
        pivot_w[i] = (phigh + plow + pclose) / 3
        r1_w[i] = 2 * pivot_w[i] - plow
        s1_w[i] = 2 * pivot_w[i] - phigh
    
    # Align weekly pivot points to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Volume average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # Start after enough data for volume MA
        # Skip if any critical data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Look for long breakout: price breaks above R1 with volume confirmation
            if (close[i] > r1_w_aligned[i] and volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Look for short breakdown: price breaks below S1 with volume confirmation
            elif (close[i] < s1_w_aligned[i] and volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot point or breakdown occurs
            if (close[i] <= pivot_w_aligned[i] or 
                (close[i] < s1_w_aligned[i] and volume_ratio > 1.5)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to pivot point or breakout occurs
            if (close[i] >= pivot_w_aligned[i] or 
                (close[i] > r1_w_aligned[i] and volume_ratio > 1.5)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1W_Pivot_Trend_v1"
timeframe = "6h"
leverage = 1.0
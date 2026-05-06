#!/usr/bin/env python3
# 6h_1wPivot_Breakout_1dTrend_Volume_v1
# Uses weekly pivot points (from 1w) as breakout levels with daily trend filter (EMA34)
# and daily volume confirmation. Designed for 6h timeframe to capture major weekly pivot breaks
# with trend alignment, working in both bull and bear markets by following the daily trend.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.

name = "6h_1wPivot_Breakout_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Weekly needs fewer bars
        return np.zeros(n)
    
    # Get daily data for trend and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: PP = (H + L + C)/3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and support/resistance levels
    pp = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Standard pivot levels: R1, S1, R2, S2
    r1 = pp + range_1w
    s1 = pp - range_1w
    r2 = pp + 2 * range_1w
    s2 = pp - 2 * range_1w
    
    # Align weekly pivot levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume filter (20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R2 with uptrend and volume
            if close[i] > r2_6h[i] and close[i] > ema_34_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with downtrend and volume
            elif close[i] < s2_6h[i] and close[i] < ema_34_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to EMA34 or breaks below S1
            if close[i] < ema_34_6h[i] or close[i] < s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to EMA34 or breaks above R1
            if close[i] > ema_34_6h[i] or close[i] > r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
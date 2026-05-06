#!/usr/bin/env python3
# 6h_WeeklyPivot_R3S3_Breakout_1dTrend_Volume
# Uses weekly pivot levels (R3/S3) as breakout levels with 1d trend filter (EMA34) and 6h volume confirmation.
# Designed for 6h timeframe to capture major weekly pivot breaks with trend alignment.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.

name = "6h_WeeklyPivot_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot levels
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    
    # Weekly R3 and S3 levels
    r3 = pp + range_w * 1.1 / 2
    s3 = pp - range_w * 1.1 / 2
    
    # Align weekly pivot levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_weekly, r3)
    s3_6h = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with uptrend and volume
            if close[i] > r3_6h[i] and close[i] > ema_34_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with downtrend and volume
            elif close[i] < s3_6h[i] and close[i] < ema_34_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to EMA34 or breaks below S3
            if close[i] < ema_34_6h[i] or close[i] < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to EMA34 or breaks above R3
            if close[i] > ema_34_6h[i] or close[i] > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf
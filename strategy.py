#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_Breakout_1wTrend_Volume
Hypothesis: Trade weekly pivot breakouts at R3/S3 levels with 1-week trend filter (EMA50) and volume confirmation.
Works in bull markets via buying R3 breakouts in uptrend and bear markets via selling S3 breakdowns in downtrend.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate WEEKLY pivot points (using previous week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Weekly pivot point (P) and support/resistance levels
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    # R3 and S3 levels (more extreme levels for breakouts)
    r3 = pivot + 2 * (prev_week_high - prev_week_low)
    s3 = pivot - 2 * (prev_week_high - prev_week_low)
    
    # Align weekly levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (approx 6 days on 6h)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly data and averages
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        pivot_val = pivot_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation AND above weekly EMA50 (uptrend)
            if close[i] > r3_val and vol_conf and close[i] > ema_50_val:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with volume confirmation AND below weekly EMA50 (downtrend)
            elif close[i] < s3_val and vol_conf and close[i] < ema_50_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly pivot (mean reversion) or below EMA50
            if close[i] < pivot_val or close[i] < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above weekly pivot or above EMA50
            if close[i] > pivot_val or close[i] > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0
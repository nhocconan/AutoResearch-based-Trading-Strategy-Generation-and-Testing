#!/usr/bin/env python3
# 12h_Pivot_R3S3_Volume_Confirmation_Reverse
# Hypothesis: Fade at 12h Pivot R3/S3 levels with volume confirmation on 12h timeframe.
# Uses 1d trend filter to avoid counter-trend trades. Target: 50-150 trades over 4 years (12-37/year).
# Works in bull/bear via 1d trend filter - only trade with the 1d trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Pivot_R3S3_Volume_Confirmation_Reverse"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 1d: Trend filter (EMA 50) ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Calculate 12h pivot levels (R3, S3) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point and range
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    
    # Align all 12h levels to 12h (self-aligned, no delay needed for same timeframe)
    r3_12h_aligned = r3_12h  # Already on 12h timeframe
    s3_12h_aligned = s3_12h  # Already on 12h timeframe
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume_12h = df_12h['volume'].values
    vol_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / np.where(vol_ma20_12h > 0, vol_ma20_12h, np.nan)
    vol_ratio_12h_aligned = vol_ratio_12h  # Already aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        r3_12h_val = r3_12h_aligned[i]
        s3_12h_val = s3_12h_aligned[i]
        vol_ratio_12h_val = vol_ratio_12h_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_val) or np.isnan(r3_12h_val) or np.isnan(s3_12h_val) or np.isnan(vol_ratio_12h_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price rejects S3 (bounces off support) with volume confirmation, only in uptrend
            if (close_val > ema_50_1d_val and  # Only long in uptrend (price above 1d EMA50)
                close_val < s3_12h_val and  # Price touched or went below S3
                prices['low'].iloc[i] <= s3_12h_val and  # Confirmed touch of S3
                close_val > s3_12h_val and  # Now bouncing back above S3
                vol_ratio_12h_val > 2.0):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price rejects R3 (bounces off resistance) with volume confirmation, only in downtrend
            elif (close_val < ema_50_1d_val and  # Only short in downtrend (price below 1d EMA50)
                  close_val > r3_12h_val and  # Price touched or went above R3
                  prices['high'].iloc[i] >= r3_12h_val and  # Confirmed touch of R3
                  close_val < r3_12h_val and  # Now falling back below R3
                  vol_ratio_12h_val > 2.0):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price reaches R3 or shows weakness
            if close_val >= r3_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price reaches S3 or shows weakness
            if close_val <= s3_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
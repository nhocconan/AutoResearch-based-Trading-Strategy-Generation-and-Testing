#!/usr/bin/env python3
# 12h_1D_Pivot_R3S3_Volume_Confirmation_v3
# Hypothesis: Fade at 12h Pivot R3/S3 levels with volume confirmation and 1d trend filter.
# Uses 1d EMA trend to avoid counter-trend trades. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear via 1d trend filter - only trade with the 1d trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1D_Pivot_R3S3_Volume_Confirmation_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 1d: EMA trend filter (34-period) ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 12h levels to 12h
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        r3_12h_val = r3_12h_aligned[i]
        s3_12h_val = s3_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_val) or np.isnan(r3_12h_val) or np.isnan(s3_12h_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price rejects S3 (bounces off support) with volume confirmation and above 1d EMA
            if (close_val < s3_12h_val and  # Price touched or went below S3
                prices['low'].iloc[i] <= s3_12h_val and  # Confirmed touch of S3
                close_val > s3_12h_val and  # Now bouncing back above S3
                vol_ratio_val > 2.0 and  # Volume confirmation
                close_val > ema_34_1d_val):  # Above 1d EMA (uptrend)
                signals[i] = 0.25
                position = 1
            # Short: Price rejects R3 (bounces off resistance) with volume confirmation and below 1d EMA
            elif (close_val > r3_12h_val and  # Price touched or went above R3
                  prices['high'].iloc[i] >= r3_12h_val and  # Confirmed touch of R3
                  close_val < r3_12h_val and  # Now falling back below R3
                  vol_ratio_val > 2.0 and  # Volume confirmation
                  close_val < ema_34_1d_val):  # Below 1d EMA (downtrend)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price reaches R3 or breaks below 1d EMA
            if close_val >= r3_12h_val or close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price reaches S3 or breaks above 1d EMA
            if close_val <= s3_12h_val or close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
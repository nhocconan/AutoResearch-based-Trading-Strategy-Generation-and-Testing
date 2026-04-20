#!/usr/bin/env python3
# 12h_1w_1d_Pivot_R3S3_Breakout_TrendFilter_V1
# Hypothesis: Breakout of 1d Pivot R3/S3 levels with volume confirmation and 1w trend filter on 12h timeframe.
# Trades only in direction of weekly trend to avoid counter-trend trades. Works in bull/bear via trend filter.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Pivot_R3S3_Breakout_TrendFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate Pivot R3/S3 levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    
    # === 1w: Trend filter (EMA34) ===
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all HTF levels to 12h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r3_1d_val = r3_1d_aligned[i]
        s3_1d_val = s3_1d_aligned[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3_1d_val) or np.isnan(s3_1d_val) or 
            np.isnan(ema_34_1w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume confirmation and weekly uptrend
            if (close_val > r3_1d_val and 
                prices['high'].iloc[i] >= r3_1d_val and  # Confirmed breakout
                vol_ratio_val > 2.0 and  # Volume confirmation
                close_val > ema_34_1w_val):  # Weekly uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume confirmation and weekly downtrend
            elif (close_val < s3_1d_val and 
                  prices['low'].iloc[i] <= s3_1d_val and  # Confirmed breakdown
                  vol_ratio_val > 2.0 and  # Volume confirmation
                  close_val < ema_34_1w_val):  # Weekly downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below S3 or shows weakness
            if close_val < s3_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above R3 or shows weakness
            if close_val > r3_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
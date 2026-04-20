#!/usr/bin/env python3
# 6h_1w_1d_Pivot_R3S3_Breakout_TrendFilter_V1
# Hypothesis: Breakout from weekly Pivot R3/S3 levels with volume confirmation,
# filtered by 1d trend (price above/below 50 EMA). Only trade breakouts in direction of 1d trend.
# Works in bull/bear via 1d trend filter - avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Pivot_R3S3_Breakout_TrendFilter_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate 1w pivot levels (R3, S3) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    
    # === 1d: 50 EMA for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all HTF levels to 6h
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r3_1w_val = r3_1w_aligned[i]
        s3_1w_val = s3_1w_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3_1w_val) or np.isnan(s3_1w_val) or 
            np.isnan(ema_50_1d_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume confirmation and 1d uptrend
            if (close_val > r3_1w_val and  # Price broke above R3
                vol_ratio_val > 1.5 and    # Volume confirmation
                close_val > ema_50_1d_val): # 1d uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume confirmation and 1d downtrend
            elif (close_val < s3_1w_val and  # Price broke below S3
                  vol_ratio_val > 1.5 and    # Volume confirmation
                  close_val < ema_50_1d_val): # 1d downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls back below R3 or trend reverses
            if close_val < r3_1w_val or close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S3 or trend reverses
            if close_val > s3_1w_val or close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
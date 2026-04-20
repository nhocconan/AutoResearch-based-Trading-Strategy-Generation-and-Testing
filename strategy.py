#!/usr/bin/env python3
# 4h_12h_Pivot_R3S3_SwingRejection_Volume
# Hypothesis: Fade at 12h Pivot R3/S3 levels with swing rejection and volume confirmation on 4h.
# Uses 12h trend filter to only trade with the 12h trend (long in uptrend, short in downtrend).
# Entry: Price rejects S3/R3 with volume > 2x average, closes back inside the pivot range.
# Exit: Price reaches opposite pivot level (R3 for long, S3 for short) or shows weakness.
# Designed for fewer trades (~20-40/year) with high win rate in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Pivot_R3S3_SwingRejection_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === Calculate 12h pivot levels (R3, S3) and pivot point ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point and range
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    
    # === 12h trend filter: EMA(34) on close ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 12h levels to 4h
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        pivot_val = pivot_12h_aligned[i]
        r3_12h_val = r3_12h_aligned[i]
        s3_12h_val = s3_12h_aligned[i]
        ema_34_val = ema_34_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(pivot_val) or np.isnan(r3_12h_val) or np.isnan(s3_12h_val) or 
            np.isnan(ema_34_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price rejects S3 (bounces off support) with volume confirmation
            # Only in 12h uptrend (price > EMA34)
            if (close_val > pivot_val and  # Price is above pivot (bullish bias)
                s3_12h_val < close_val <= s3_12h_val * 1.005 and  # Price near S3 (within 0.5%)
                ema_34_val < close_val and  # 12h uptrend filter
                vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price rejects R3 (bounces off resistance) with volume confirmation
            # Only in 12h downtrend (price < EMA34)
            elif (close_val < pivot_val and  # Price is below pivot (bearish bias)
                  r3_12h_val * 0.995 <= close_val < r3_12h_val and  # Price near R3 (within 0.5%)
                  ema_34_val > close_val and  # 12h downtrend filter
                  vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price reaches R3 or shows weakness (breaks below pivot)
            if close_val >= r3_12h_val or close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price reaches S3 or shows weakness (breaks above pivot)
            if close_val <= s3_12h_val or close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
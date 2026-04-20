#!/usr/bin/env python3
# 4h_1d_Pivot_R3S3_Reversal_Volume
# Hypothesis: Mean reversion at daily Camarilla R3/S3 levels with volume confirmation on 4h.
# Uses 1d trend filter to avoid counter-trend trades. Works in bull/bear via trend filter.
# Target: 15-40 trades/year (60-160 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_R3S3_Reversal_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate 1d pivot levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    
    # === 1d trend filter: EMA34 > EMA89 = uptrend, else downtrend ===
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = close_1d_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    uptrend_1d = ema34_1d > ema89_1d
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 1d levels and trend to 4h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r3_1d_val = r3_1d_aligned[i]
        s3_1d_val = s3_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        uptrend_val = uptrend_1d_aligned[i] > 0.5  # True if uptrend
        
        # Skip if any value is NaN
        if (np.isnan(r3_1d_val) or np.isnan(s3_1d_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price rejects S3 (bounces off support) in uptrend with volume confirmation
            if (close_val < s3_1d_val and  # Price touched or went below S3
                prices['low'].iloc[i] <= s3_1d_val and  # Confirmed touch of S3
                close_val > s3_1d_val and  # Now bouncing back above S3
                vol_ratio_val > 1.8 and  # Volume confirmation
                uptrend_val):  # Only long in uptrend
                signals[i] = 0.25
                position = 1
            # Short: Price rejects R3 (bounces off resistance) in downtrend with volume confirmation
            elif (close_val > r3_1d_val and  # Price touched or went above R3
                  prices['high'].iloc[i] >= r3_1d_val and  # Confirmed touch of R3
                  close_val < r3_1d_val and  # Now falling back below R3
                  vol_ratio_val > 1.8 and  # Volume confirmation
                  not uptrend_val):  # Only short in downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price reaches R3 or shows weakness
            if close_val >= r3_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price reaches S3 or shows weakness
            if close_val <= s3_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
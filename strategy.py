#!/usr/bin/env python3
# 4h_1d_Pivot_Strategy_With_Trend_and_Volume
# Hypothesis: Daily pivot levels act as strong support/resistance. Long when price breaks above daily R1 with volume and above 4h EMA50 (uptrend), short when breaks below daily S1 with volume and below 4h EMA50 (downtrend). Uses 1d timeframe for pivots and 4h for execution. Designed to work in both bull and bear markets by following the 4h EMA trend filter.
# Target: 20-40 trades per year per symbol to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_Strategy_With_Trend_and_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H+L+C)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Daily R1 and S1 levels
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    
    # Calculate 4h EMA50 for trend filter
    close = prices['close'].values
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume ratio (current vs 20-period average)
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align daily levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = close[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        ema50_val = ema50[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema50_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume confirmation and above EMA50 (uptrend)
            if (close_val > r1_val and vol_ratio_val > 2.0 and close_val > ema50_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume confirmation and below EMA50 (downtrend)
            elif (close_val < s1_val and vol_ratio_val > 2.0 and close_val < ema50_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below daily pivot point
            if close_val <= pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above daily pivot point
            if close_val >= pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
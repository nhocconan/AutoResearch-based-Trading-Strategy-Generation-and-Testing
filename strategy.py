#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: Camarilla pivot levels on 1d with volume confirmation and daily trend filter captures institutional supply/demand zones. Works in bull markets (buying at support) and bear markets (selling at resistance) by trading with daily trend. Targets 12-37 trades/year by requiring 1d pivot touch + volume spike + daily trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4
    # R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align 1d levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        vol_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S1 OR trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R1 OR trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches S3 + volume + uptrend
            if (abs(close[i] - s3_aligned[i]) < 0.001 * close[i] and 
                vol_confirm and 
                close[i] > ema50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 + volume + downtrend
            elif (abs(close[i] - r3_aligned[i]) < 0.001 * close[i] and 
                  vol_confirm and 
                  close[i] < ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
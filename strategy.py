#!/usr/bin/env python3
"""
12h Camarilla Pivot + 1d Trend + Volume Filter
Hypothesis: Camarilla pivot levels on 12h provide high-probability reversal zones when aligned with daily trend and volume confirmation. Works in both bull and bear by taking longs in uptrends and shorts in downtrends. Targets 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Camarilla Pivot levels (based on previous day)
    # Calculate daily high/low/close for pivot
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    pivot = (daily_high + daily_low + daily_close) / 3
    range_val = daily_high - daily_low
    
    # Camarilla levels: S1, S2, S3, R1, R2, R3
    s1 = close_pivot = pivot - (range_val * 1.0 / 6)
    s2 = pivot - (range_val * 2.0 / 6)
    s3 = pivot - (range_val * 3.0 / 6)
    r1 = pivot + (range_val * 1.0 / 6)
    r2 = pivot + (range_val * 2.0 / 6)
    r3 = pivot + (range_val * 3.0 / 6)
    
    # Align pivot levels to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume filter (>1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_filter[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (strong support broken) OR trend reverses
            if (close[i] <= s3_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (strong resistance broken) OR trend reverses
            if (close[i] >= r3_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry at S3 with trend alignment and volume
            if (close[i] <= s3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry at R3 with trend alignment and volume
            elif (close[i] >= r3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
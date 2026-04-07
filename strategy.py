#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Camarilla pivot levels on weekly timeframe identify key support/resistance zones, 
while 12h timeframe provides entry signals with volume confirmation and trend filter. 
Works in both bull and bear markets by fading extremes during ranging conditions and 
following breakouts during trending periods. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
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
    
    # Weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (based on previous week)
    # Typical price = (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Camarilla levels based on previous week's range
    high_prev = df_1w['high'].shift(1)
    low_prev = df_1w['low'].shift(1)
    close_prev = df_1w['close'].shift(1)
    
    # Pivot point
    pivot = (high_prev + low_prev + close_prev) / 3
    # Range
    range_val = high_prev - low_prev
    
    # Camarilla levels
    # Resistance levels
    r1 = close_prev + (range_val * 1.1 / 12)
    r2 = close_prev + (range_val * 1.1 / 6)
    r3 = close_prev + (range_val * 1.1 / 4)
    r4 = close_prev + (range_val * 1.1 / 2)
    # Support levels
    s1 = close_prev - (range_val * 1.1 / 12)
    s2 = close_prev - (range_val * 1.1 / 6)
    s3 = close_prev - (range_val * 1.1 / 4)
    s4 = close_prev - (range_val * 1.1 / 2)
    
    # Trend filter: 50-period EMA on weekly
    ema_50 = df_1w['close'].ewm(span=50, adjust=False).mean()
    
    # Align all weekly data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50.values)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 or trend turns bearish
            if close[i] >= r3_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches S3 or trend turns bullish
            if close[i] <= s3_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price bounces off S1/S2 with volume and bullish trend
            if (vol_confirm and close[i] > ema_50_aligned[i] and
                (close[i] <= s1_aligned[i] * 1.005 or close[i] <= s2_aligned[i] * 1.005)):
                position = 1
                signals[i] = 0.25
            # Short entry: price rejects R1/R2 with volume and bearish trend
            elif (vol_confirm and close[i] < ema_50_aligned[i] and
                  (close[i] >= r1_aligned[i] * 0.995 or close[i] >= r2_aligned[i] * 0.995)):
                position = -1
                signals[i] = -0.25
    
    return signals
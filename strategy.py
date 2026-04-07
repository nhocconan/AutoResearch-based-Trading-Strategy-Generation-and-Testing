#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use weekly Camarilla pivot levels for entry, filtered by 1d EMA trend and volume confirmation.
- In uptrend (price > 1d EMA50): long near S3 (weekly support) or break above R4 with volume
- In downtrend (price < 1d EMA50): short near R3 (weekly resistance) or break below S4 with volume
Weekly pivots provide stronger support/resistance than daily, reducing false signals.
Volume confirms genuine tests. Target: 12-37 trades/year (~50-150 total over 4 years).
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # Weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous week (to avoid look-ahead)
    prev_week_close = df_1w['close'].shift(1)
    prev_week_high = df_1w['high'].shift(1)
    prev_week_low = df_1w['low'].shift(1)
    
    # Pivot point
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    range_val = prev_week_high - prev_week_low
    
    # Camarilla levels (using weekly range)
    # Resistance levels
    r1 = pivot + (range_val * 1.1 / 12)
    r2 = pivot + (range_val * 1.1 / 6)
    r3 = pivot + (range_val * 1.1 / 4)
    r4 = pivot + (range_val * 1.1 / 2)
    # Support levels
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    
    # Align all levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)  # realign for safety
    
    # Volume confirmation (24-period average on 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S4 (breakdown) or trend turns bearish
            if close[i] < s4_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R4 (breakout) or trend turns bullish
            if close[i] > r4_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price tests S3 with volume in uptrend
            if (close[i] <= s3_aligned[i] * 1.005 and close[i] >= s3_aligned[i] * 0.995 and  # near S3
                vol_confirm and 
                close[i] > ema_50_aligned[i]):  # uptrend filter
                position = 1
                signals[i] = 0.25
            # Short entry: price tests R3 with volume in downtrend
            elif (close[i] >= r3_aligned[i] * 0.995 and close[i] <= r3_aligned[i] * 1.005 and  # near R3
                  vol_confirm and 
                  close[i] < ema_50_aligned[i]):  # downtrend filter
                position = -1
                signals[i] = -0.25
            # Long breakout: price breaks above R4 with volume in uptrend
            elif (close[i] > r4_aligned[i] and
                  vol_confirm and 
                  close[i] > ema_50_aligned[i]):  # uptrend filter
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below S4 with volume in downtrend
            elif (close[i] < s4_aligned[i] and
                  vol_confirm and 
                  close[i] < ema_50_aligned[i]):  # downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals
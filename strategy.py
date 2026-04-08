#!/usr/bin/env python3
"""
6H Weekly Pivot + Daily Trend + Volume Confirmation v1
Hypothesis: Weekly pivot points provide strong support/resistance levels. 
Price breaking above weekly R3 with daily uptrend and volume confirmation indicates strong momentum.
Price breaking below weekly S3 with daily downtrend and volume confirmation indicates strong weakness.
Uses 6h timeframe to capture multi-day moves while avoiding excessive noise.
Target: 15-35 trades/year per signal.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_daily_trend_volume_v1"
timeframe = "6h"
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
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot points (using prior week's data)
    weekly_high = df_1w['high'].shift(1)
    weekly_low = df_1w['low'].shift(1)
    weekly_close = df_1w['close'].shift(1)
    
    # Calculate pivot levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r4 = weekly_high + 3 * (pivot - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - pivot)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(21) for trend filter
    ema_21 = df_1d['close'].ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1w, r3.values)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3.values)
    r4_6h = align_htf_to_ltf(prices, df_1w, r4.values)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4.values)
    ema_21_6h = align_htf_to_ltf(prices, df_1d, ema_21.values)
    
    # Volume filter (>1.5x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(ema_21_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly S3 or trend reverses
            if close[i] <= s3_6h[i] or close[i] < ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly R3 or trend reverses
            if close[i] >= r3_6h[i] or close[i] > ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long at weekly R3 with trend alignment
            if (close[i] >= r3_6h[i] and 
                close[i] > ema_21_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at weekly S3 with trend alignment
            elif (close[i] <= s3_6h[i] and 
                  close[i] < ema_21_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
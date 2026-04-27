#!/usr/bin/env python3
"""
#101007 - 6h_WeeklyPivot_Direction_DailyBreakout_Volume
Hypothesis: Use weekly pivot points from 1w data to establish long-term bias, then take daily breakouts with volume confirmation on 6h timeframe.
In weekly uptrend (price above weekly pivot), look for long entries on daily resistance breakouts with volume.
In weekly downtrend (price below weekly pivot), look for short entries on daily support breakouts with volume.
This combines long-term trend bias with short-term momentum, reducing false signals in choppy markets.
Target: 12-25 trades/year to minimize fee drag on 6s timeframe. Uses discrete position sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (long-term bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivot data to 6h timeframe
    pivot_1w_aligned = align_ltf_to_htf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_ltf_to_htf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_ltf_to_htf(prices, df_1w, s1_1w)
    
    # Get daily data for breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily high/low for breakout detection
    daily_high = high_1d
    daily_low = low_1d
    
    # Align daily data to 6h timeframe
    daily_high_aligned = align_ltf_to_htf(prices, df_1d, daily_high)
    daily_low_aligned = align_ltf_to_htf(prices, df_1d, daily_low)
    
    # Volume filter: volume > 2.0x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend bias
        weekly_uptrend = close[i] > pivot_1w_aligned[i]
        weekly_downtrend = close[i] < pivot_1w_aligned[i]
        
        # Long condition: weekly uptrend + price breaks above daily high + volume
        if (weekly_uptrend and 
            close[i] > daily_high_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: weekly downtrend + price breaks below daily low + volume
        elif (weekly_downtrend and 
              close[i] < daily_low_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to weekly pivot (mean reversion to bias)
        elif position == 1 and close[i] < pivot_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > pivot_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Direction_DailyBreakout_Volume"
timeframe = "6h"
leverage = 1.0
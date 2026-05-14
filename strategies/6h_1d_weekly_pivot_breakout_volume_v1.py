#!/usr/bin/env python3
"""
6h_1d_weekly_pivot_breakout_volume_v1
Hypothesis: Use 6h price action with 1d/1w pivot levels and volume confirmation.
Long when price breaks above weekly pivot resistance with bullish daily bias and volume.
Short when price breaks below weekly pivot support with bearish daily bias and volume.
Designed to capture institutional breakouts at key weekly levels with trend alignment.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) by requiring strong breakouts at pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_breakout_volume_v1"
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
    
    # Get 6h data for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Get 1d data for daily bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    pivot_point = (high_1w + low_1w + close_1w) / 3.0
    # Weekly resistance and support levels
    r1 = 2 * pivot_point - low_1w
    s1 = 2 * pivot_point - high_1w
    r2 = pivot_point + (high_1w - low_1w)
    s2 = pivot_point - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot_point - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot_point)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily bias: 1d close vs 1d open (bullish if close > open)
    daily_bias = df_1d['close'].values > df_1d['open'].values
    daily_bias_aligned = align_htf_to_ltf(prices, df_1d, daily_bias.astype(float))
    
    # Volume confirmation: volume > 1.3x average of last 10 periods
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 10
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(daily_bias_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly S1 or daily bias turns bearish
            if close[i] < s1_aligned[i] or daily_bias_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly R1 or daily bias turns bullish
            if close[i] > r1_aligned[i] or daily_bias_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly R1 with bullish daily bias and volume
            if close[i] > r1_aligned[i] and daily_bias_aligned[i] > 0.5 and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly S1 with bearish daily bias and volume
            elif close[i] < s1_aligned[i] and daily_bias_aligned[i] < 0.5 and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
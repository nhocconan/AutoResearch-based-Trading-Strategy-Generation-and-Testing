#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_1d_trend_volume_v1
# Hypothesis: Uses weekly pivot levels with 1d trend filter and volume confirmation on 6h timeframe.
# Goes long when price breaks above weekly R1 in daily uptrend with volume confirmation.
# Goes short when price breaks below weekly S1 in daily downtrend with volume confirmation.
# Uses 6h candles to reduce trade frequency and avoid fee drag. Target: 12-37 trades/year.
# Weekly pivots provide strong support/resistance; trend filter avoids counter-trend trades; volume confirms breakout strength.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_1d_trend_volume_v1"
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
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot_1w - low_1w
    s1 = 2 * pivot_1w - high_1w
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and daily data to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation on 6h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema50_1d[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price falls below S1 or trend changes to downtrend
            if close[i] <= s1_aligned[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above R1 or trend changes to uptrend
            if close[i] >= r1_aligned[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long entry: price breaks above R1 in uptrend
                if daily_uptrend and close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below S1 in downtrend
                elif daily_downtrend and close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
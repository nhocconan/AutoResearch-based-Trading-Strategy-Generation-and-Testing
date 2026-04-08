#!/usr/bin/env python3
# 6h_weekly_pivot_breakout
# Hypothesis: Breakouts above weekly R3 or below weekly S3 pivots with volume confirmation and price > EMA200 (1d) for trend alignment.
# Weekly pivots provide institutional support/resistance levels. Breakouts indicate momentum shifts.
# Volume > 1.5x average filters weak moves. EMA200 filter ensures trades align with higher timeframe trend.
# Designed to work in both bull (breakouts above R3) and bear (breakdowns below S3) markets.
# Target: 20-40 trades/year (~80-160 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points and support/resistance levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate average volume for confirmation (30-period)
    avg_volume = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly pivot OR trend turns against us
            if (close[i] < pivot_1w_aligned[i]) or (close[i] < ema_200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly pivot OR trend turns against us
            if (close[i] > pivot_1w_aligned[i]) or (close[i] > ema_200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above weekly R3 with uptrend and volume confirmation
            if (close[i] > r3_1w_aligned[i]) and (close[i] > ema_200_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly S3 with downtrend and volume confirmation
            elif (close[i] < s3_1w_aligned[i]) and (close[i] < ema_200_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals
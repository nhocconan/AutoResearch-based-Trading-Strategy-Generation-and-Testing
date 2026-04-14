#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d/1w pivot confluence and volume confirmation
# Uses 1d/1w pivot levels (R3/S3 as fade points, R4/S4 as breakout continuation)
# Breakouts only in direction of higher timeframe pivot bias (bullish if above weekly pivot)
# Volume > 1.5x 20-period average confirms institutional participation
# Aims for 15-30 trades/year (60-120 total over 4 years) with 0.25 position size

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE for pivot levels and trend bias
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 1 or len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1d pivot points (standard floor trader pivots)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    r4_1d = r3_1d + (high_1d - low_1d)
    s4_1d = s3_1d - (high_1d - low_1d)
    
    # Calculate 1w pivot for trend bias
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    r4_1w = r3_1w + (high_1w - low_1w)
    s4_1w = s3_1w - (high_1w - low_1w)
    
    # Align HTF pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Donchian channel (20 periods) on 6h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(pivot_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or
            np.isnan(r4_1w_aligned[i]) or
            np.isnan(s4_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market bias from weekly pivot
        bullish_bias = close[i] > pivot_1w_aligned[i]
        bearish_bias = close[i] < pivot_1w_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Breakout above R4 (1d or 1w) with bullish bias + volume
            if ( (close[i] > dc_upper[i] and 
                  (close[i] > r4_1d_aligned[i] or close[i] > r4_1w_aligned[i]) and
                  bullish_bias and 
                  volume_confirmed) ):
                position = 1
                signals[i] = position_size
            # Enter short: Breakdown below S4 (1d or 1w) with bearish bias + volume
            elif ( (close[i] < dc_lower[i] and 
                    (close[i] < s4_1d_aligned[i] or close[i] < s4_1w_aligned[i]) and
                    bearish_bias and 
                    volume_confirmed) ):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns to 1d pivot or breaks below S3 (fade signal)
            if close[i] < pivot_1d_aligned[i] or close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price returns to 1d pivot or breaks above R3 (fade signal)
            if close[i] > pivot_1d_aligned[i] or close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d1w_Pivot_Confluence_Donchian_Volume_v1"
timeframe = "6h"
leverage = 1.0
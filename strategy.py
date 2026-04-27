#!/usr/bin/env python3
"""
#100811 - 6h_Donchian20_WeeklyPivot_Direction_Volume
Hypothesis: Combine 6h Donchian breakout with weekly pivot direction and volume confirmation.
In bull markets: breakouts above weekly pivot resistance with volume. In bear markets: breakouts below weekly pivot support with volume.
Weekly pivot provides structural support/resistance that works across market cycles. Volume confirms institutional interest.
Target: 15-30 trades/year to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Get 1d data for additional context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous week
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Weekly pivot point and support/resistance
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    weekly_r1 = weekly_pivot + (weekly_range * 1.1) / 2  # R1
    weekly_s1 = weekly_pivot - (weekly_range * 1.1) / 2  # S1
    weekly_r2 = weekly_pivot + weekly_range  # R2
    weekly_s2 = weekly_pivot - weekly_range  # S2
    
    # Align weekly levels to 6h timeframe (previous week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma = np.full(n, np.nan)
    for i in range(30 - 1, n):
        vol_ma[i] = np.mean(volume[i - 30 + 1:i + 1])
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: breakout above resistance with volume
        # Primary: break above weekly R2 (strong bullish)
        # Secondary: break above weekly R1 with price above pivot
        long_signal = False
        if (high[i] > r2_aligned[i] and volume_filter[i]):
            long_signal = True
        elif (high[i] > r1_aligned[i] and close[i] > pivot_aligned[i] and volume_filter[i]):
            long_signal = True
            
        # Short conditions: breakdown below support with volume
        # Primary: break below weekly S2 (strong bearish)
        # Secondary: break below weekly S1 with price below pivot
        short_signal = False
        if (low[i] < s2_aligned[i] and volume_filter[i]):
            short_signal = True
        elif (low[i] < s1_aligned[i] and close[i] < pivot_aligned[i] and volume_filter[i]):
            short_signal = True
        
        # Entry logic
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit logic: return to weekly pivot (mean reversion)
        elif position == 1 and close[i] < pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > pivot_aligned[i]:
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

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0
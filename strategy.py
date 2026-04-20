#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_With_Trend_Filter
Hypothesis: Trade daily price breakouts above/below weekly pivot resistance/support levels with volume confirmation and weekly trend filter.
Long when price breaks above weekly R1 with volume spike and weekly uptrend; short when breaks below weekly S1 with volume spike and weekly downtrend.
Uses weekly pivot levels (calculated from prior weekly bar) and volume > 1.5x 20-period average for confirmation.
Designed for 1d timeframe to capture medium-term moves while reducing noise. Target: 20-50 total trades over 4 years (5-12/year).
Works in bull/bear: weekly trend filter avoids counter-trend trades, volume filter reduces false breakouts.
"""

name = "1d_WeeklyPivot_R1S1_Breakout_With_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior weekly bar's high, low, close)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Pivot point calculation: PP = (H + L + C) / 3
    # R1 = 2*PP - L, S1 = 2*PP - H
    pp_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r1_weekly = 2 * pp_weekly - low_weekly
    s1_weekly = 2 * pp_weekly - high_weekly
    
    # Align weekly pivot levels to 1d timeframe (already delayed by one bar via align_htf_to_ltf)
    pp_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Calculate volume filter (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Ensure indicators are ready (20 for volume MA + buffer)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_weekly_aligned[i]) or np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume filter AND weekly uptrend (close > weekly pivot)
            if close[i] > r1_weekly_aligned[i] and volume_filter[i] and close[i] > pp_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume filter AND weekly downtrend (close < weekly pivot)
            elif close[i] < s1_weekly_aligned[i] and volume_filter[i] and close[i] < pp_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly pivot point
            if close[i] < pp_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly pivot point
            if close[i] > pp_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
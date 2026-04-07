#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Pivot Range Breakout with Volume Confirmation
# Hypothesis: Weekly pivot ranges define institutional support/resistance.
# Breakouts above weekly R1 or below weekly S1 with volume confirmation
# capture momentum moves. Weekly context filters noise, volume reduces false breakouts.
# Works in bull/bear: breakouts in either direction capture trends.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "6h_weekly_pivot_range_breakout_volume_v1"
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
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Support 1 = (2 * P) - H
    s1 = (2 * pivot) - weekly_high
    # Resistance 1 = (2 * P) - L
    r1 = (2 * pivot) - weekly_low
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    
    # Volume filter on 6h: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below weekly pivot or volume dries up
            if close[i] < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above weekly pivot or volume dries up
            if close[i] > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: break above weekly R1 with volume
            if (high[i] > r1_aligned[i] and close[i] > r1_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: break below weekly S1 with volume
            elif (low[i] < s1_aligned[i] and close[i] < s1_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
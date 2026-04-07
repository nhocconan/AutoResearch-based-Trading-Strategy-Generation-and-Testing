#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Pivot + Volume Breakout
# Hypothesis: Trade breakouts of weekly pivot levels (R1/S1) with volume confirmation.
# Weekly pivots provide structural support/resistance. Volume confirms institutional interest.
# Works in bull/bear by trading breakouts in either direction with volume filter.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_weekly_pivot_volume_v1"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Weekly pivot: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivots to 6h (shifted by 1 week for lookback)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Volume filter: 20-period volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly pivot
            if close[i] < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above weekly pivot
            if close[i] > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_confirm:
                # Breakout above R1
                if close[i] > r1_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below S1
                elif close[i] < s1_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
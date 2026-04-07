#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Pivot Reversion with Volume Filter
# Hypothesis: Weekly pivot levels act as strong support/resistance. 
# Price reversing from weekly S1/R1 with volume confirmation indicates 
# institutional defense of levels, leading to mean reversion.
# Works in both bull and bear:
# - In bull: price pulls back to weekly S1 with volume = long opportunity
# - In bear: price rallies to weekly R1 with volume = short opportunity
# Uses volume filter to confirm institutional participation at key levels.
# Target: 15-25 trades/year (60-100 over 4 years).

name = "6h_weekly_pivot_reversion_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly data (previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    prev_weekly_close[0] = prev_weekly_close[1] if len(prev_weekly_close) > 1 else 0
    
    # Calculate weekly pivot points
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r1 = (2 * weekly_pivot) - prev_weekly_low
    weekly_s1 = (2 * weekly_pivot) - prev_weekly_high
    weekly_r2 = weekly_pivot + (prev_weekly_high - prev_weekly_low)
    weekly_s2 = weekly_pivot - (prev_weekly_high - prev_weekly_low)
    
    # Align to 6h timeframe (use previous week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # Volume filter: volume > 1.3x 26-period average (slightly tighter for 6h)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=26, min_periods=26).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(26, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches weekly pivot or volume drops
            if close[i] >= pivot_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price reaches weekly pivot or volume drops
            if close[i] <= pivot_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price touches/slightly penetrates S1 with volume (defense of support)
            if low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price touches/slightly penetrates R1 with volume (defense of resistance)
            elif high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
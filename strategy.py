#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1D Weekly Pivot Breakout with Volume Confirmation
# Hypothesis: Weekly pivot levels (R1/S1) act as strong support/resistance in the 1D timeframe.
# Price breaking through these levels with volume confirmation indicates institutional breakout.
# Works in both bull and bear markets:
# - In bull: price breaks above R1 and continues up (momentum)
# - In bear: price breaks below S1 and continues down (momentum)
# Target: 20-50 trades/year (80-200 over 4 years).

name = "1d_weekly_pivot_breakout_volume_v1"
timeframe = "1d"
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
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels: based on previous week's range
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Weekly pivot formulas (based on previous week)
    # R2 = Pivot + (High - Low)
    # R1 = (2 * Pivot) - Low
    # Pivot = (High + Low + Close) / 3
    # S1 = (2 * Pivot) - High
    # S2 = Pivot - (High - Low)
    
    # Calculate for previous week (shift by 1 to avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    prev_weekly_close[0] = prev_weekly_close[1] if len(prev_weekly_close) > 1 else 0
    
    # Calculate weekly pivot levels for previous week
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    r1 = (2 * pivot) - prev_weekly_low
    s1 = (2 * pivot) - prev_weekly_high
    
    # Align to 1d timeframe (use previous week's levels)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below pivot or volume drops
            if close[i] < pivot[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above pivot or volume drops
            if close[i] > pivot[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R1 with volume
            if close[i] > r1_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S1 with volume
            elif close[i] < s1_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Pivot Reversal with Volume Confirmation
# Hypothesis: Daily pivot levels act as strong support/resistance. 
# Price reversing off S1 with volume indicates buying pressure, leading to bounce.
# Price reversing off R1 with volume indicates selling pressure, leading to pullback.
# Works in both bull and bear:
# - In bull: reversals off S1 continue uptrend; reversals off R1 create pullbacks in uptrend
# - In bear: reversals off R1 continue downtrend; reversals off S1 create bounces in downtrend
# Uses volume filter to confirm institutional participation at key levels.
# Target: 15-35 trades/year (60-140 over 4 years).

name = "4h_daily_pivot_reversal_volume_v1"
timeframe = "4h"
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
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily data (previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high[0] = prev_daily_high[1] if len(prev_daily_high) > 1 else 0
    prev_daily_low[0] = prev_daily_low[1] if len(prev_daily_low) > 1 else 0
    prev_daily_close[0] = prev_daily_close[1] if len(prev_daily_close) > 1 else 0
    
    # Calculate daily pivot points
    # Pivot = (High + Low + Close) / 3
    # R1 = (2 * Pivot) - Low
    # S1 = (2 * Pivot) - High
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_r1 = (2 * daily_pivot) - prev_daily_low
    daily_s1 = (2 * daily_pivot) - prev_daily_high
    
    # Align to 4h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below S1 or volume drops
            if low[i] < s1_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above R1 or volume drops
            if high[i] > r1_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price bounces off S1 with volume (close above S1 after touching/test)
            if low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price rejects at R1 with volume (close below R1 after touching/test)
            elif high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h/1d Pivot Breakout with Volume Filter (Tightened)
# Hypothesis: Daily and 12h pivot levels act as strong support/resistance.
# Price breaking above R1 with volume indicates institutional buying, leading to continuation.
# Price breaking below S1 with volume indicates institutional selling, leading to continuation.
# Works in both bull and bear markets because: In bull, breaks above R1 continue up; breaks below S1 get bought (mean reversion).
# In bear, breaks below S1 continue down; breaks above R1 get sold (mean reversion).
# Volume filter ensures only institutional participation triggers entries.
# Target: 12-37 trades/year (50-150 over 4 hours). Reduced frequency via stricter entry conditions.

name = "12h_1d_pivot_breakout_volume_v1"
timeframe = "12h"
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
    
    # Get daily and 12h data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_daily) < 2 or len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate daily data (previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate 12h data (previous 12h bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Shift by 1 to use previous day's/12h bar's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high[0] = prev_daily_high[1] if len(prev_daily_high) > 1 else 0
    prev_daily_low[0] = prev_daily_low[1] if len(prev_daily_low) > 1 else 0
    prev_daily_close[0] = prev_daily_close[1] if len(prev_daily_close) > 1 else 0
    
    prev_12h_high = np.roll(high_12h, 1)
    prev_12h_low = np.roll(low_12h, 1)
    prev_12h_close = np.roll(close_12h, 1)
    prev_12h_high[0] = prev_12h_high[1] if len(prev_12h_high) > 1 else 0
    prev_12h_low[0] = prev_12h_low[1] if len(prev_12h_low) > 1 else 0
    prev_12h_close[0] = prev_12h_close[1] if len(prev_12h_close) > 1 else 0
    
    # Calculate daily pivot points
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_r1 = (2 * daily_pivot) - prev_daily_low
    daily_s1 = (2 * daily_pivot) - prev_daily_high
    daily_r2 = daily_pivot + (prev_daily_high - prev_daily_low)
    daily_s2 = daily_pivot - (prev_daily_high - prev_daily_low)
    
    # Calculate 12h pivot points
    pivot_12h = (prev_12h_high + prev_12h_low + prev_12h_close) / 3.0
    r1_12h = (2 * pivot_12h) - prev_12h_low
    s1_12h = (2 * pivot_12h) - prev_12h_high
    r2_12h = pivot_12h + (prev_12h_high - prev_12h_low)
    s2_12h = pivot_12h - (prev_12h_high - prev_12h_low)
    
    # Align to 12h timeframe (use previous day's/12h bar's levels)
    daily_pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    daily_r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    daily_r2_aligned = align_htf_to_ltf(prices, df_daily, daily_r2)
    daily_s2_aligned = align_htf_to_ltf(prices, df_daily, daily_s2)
    
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(daily_pivot_aligned[i]) or np.isnan(daily_r1_aligned[i]) or 
            np.isnan(daily_s1_aligned[i]) or np.isnan(daily_r2_aligned[i]) or 
            np.isnan(daily_s2_aligned[i]) or np.isnan(pivot_12h_aligned[i]) or 
            np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(r2_12h_aligned[i]) or np.isnan(s2_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to daily pivot or 12h pivot or volume drops
            if (close[i] <= daily_pivot_aligned[i] or close[i] <= pivot_12h_aligned[i] or 
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to daily pivot or 12h pivot or volume drops
            if (close[i] >= daily_pivot_aligned[i] or close[i] >= pivot_12h_aligned[i] or 
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above daily R1 or 12h R1 with volume
            if ((high[i] > daily_r1_aligned[i] or high[i] > r1_12h_aligned[i]) and 
                (close[i] > daily_r1_aligned[i] or close[i] > r1_12h_aligned[i]) and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below daily S1 or 12h S1 with volume
            elif ((low[i] < daily_s1_aligned[i] or low[i] < s1_12h_aligned[i]) and 
                  (close[i] < daily_s1_aligned[i] or close[i] < s1_12h_aligned[i]) and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
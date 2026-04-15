#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h pivot breakout with volume confirmation and session filter
# Uses 1d pivots for direction, 1h for entry timing. Session filter (08-20 UTC) reduces noise.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
# Works in bull (breakouts) and bear (rejections at S1/R1) via symmetric long/short logic.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels (HTF)
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily pivot levels
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    r2 = pivot + (daily_high - daily_low)
    s2 = pivot - (daily_high - daily_low)
    
    # Align pivot levels to 1h timeframe
    pivot_aligned = align_htf_to_ltf(prices, daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, daily, r1)
    s1_aligned = align_htf_to_ltf(prices, daily, s1)
    r2_aligned = align_htf_to_ltf(prices, daily, r2)
    s2_aligned = align_htf_to_ltf(prices, daily, s2)
    
    # Volume filter: current 1h volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Range filter: avoid trading when price is within 0.3% of pivot
    price_to_pivot = np.abs(close - pivot_aligned) / pivot_aligned
    range_filter = price_to_pivot > 0.003
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when all filters pass
        if volume_filter[i] and range_filter[i] and session_filter[i]:
            # Long conditions: price breaks above R2 with volume (strong breakout)
            if close[i] > r2_aligned[i]:
                signals[i] = 0.20
            # Long conditions: price bounces from S1 with volume (above S1, below S2)
            elif close[i] > s1_aligned[i] and close[i] < s2_aligned[i]:
                signals[i] = 0.20
            # Short conditions: price breaks below S2 with volume (strong breakout)
            elif close[i] < s2_aligned[i]:
                signals[i] = -0.20
            # Short conditions: price rejected at R1 with volume (below R1, above R2)
            elif close[i] < r1_aligned[i] and close[i] > r2_aligned[i]:
                signals[i] = -0.20
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_Pivot_R1_S1_R2_S2_Breakout_Volume_RangeFilter_Session"
timeframe = "1h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels
    daily = get_htf_data(prices, '1d')
    
    # Calculate daily pivot levels (classic floor trader pivots)
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    r2 = pivot + (daily_high - daily_low)
    s2 = pivot - (daily_high - daily_low)
    r3 = daily_high + 2 * (pivot - daily_low)
    s3 = daily_low - 2 * (daily_high - pivot)
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, daily, r1)
    s1_aligned = align_htf_to_ltf(prices, daily, s1)
    r2_aligned = align_htf_to_ltf(prices, daily, r2)
    s2_aligned = align_htf_to_ltf(prices, daily, s2)
    r3_aligned = align_htf_to_ltf(prices, daily, r3)
    s3_aligned = align_htf_to_ltf(prices, daily, s3)
    
    # Volume filter: current 12h volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above R1 with volume, or bounces from S1 with volume
        if volume_filter[i]:
            if close[i] > r1_aligned[i]:
                signals[i] = 0.25
            elif close[i] < s1_aligned[i] and close[i] > s2_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below S1 with volume, or rejected at R1 with volume
            elif close[i] < s1_aligned[i]:
                signals[i] = -0.25
            elif close[i] > r1_aligned[i] and close[i] < r2_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0
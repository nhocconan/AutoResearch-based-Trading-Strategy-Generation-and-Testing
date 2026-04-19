#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d pivot point reversal strategy.
# Uses daily pivot points (R1, S1) for mean reversion entries and R4/S4 for breakout continuations.
# Combines with volume confirmation and session filter (08-20 UTC) to avoid low-volume noise.
# Works in both bull and bear markets by adapting to pivot levels.
# Targets 20-40 trades/year (80-160 total over 4 years) with strict entry conditions.

name = "6h_1d_Pivot_R1S1_R4S4_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for pivot points (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: P = (H+L+C)/3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Support and resistance levels
    s1 = 2 * pivot - high_1d
    r1 = 2 * pivot - low_1d
    s2 = pivot - (high_1d - low_1d)
    r2 = pivot + (high_1d - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    r3 = high_1d + 2 * (pivot - low_1d)
    s4 = s3 - (high_1d - low_1d)
    r4 = r3 + (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long reversal at S1: price touches/bounces off S1 with volume
            if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short reversal at R1: price touches/rejects at R1 with volume
            elif (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            # Long breakout continuation above R4 with volume
            elif (close[i] > r4_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown continuation below S4 with volume
            elif (close[i] < s4_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price reaches R1 (take profit) or breaks below S1 (stop)
            if close[i] >= r1_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price reaches S1 (take profit) or breaks above R1 (stop)
            if close[i] <= s1_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot from 1d + Volume Spike
# Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
# Fade at R3/S3 levels (mean reversion), breakout continuation at R4/S4 levels.
# Volume spikes confirm institutional participation. Works in both bull and bear markets:
# - In bull: buy dips to S3/S4 with volume, sell rallies to R3/R4 with volume
# - In bear: sell rallies to R3/R4 with volume, buy dips to S3/S4 with volume
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_camarilla_pivot_1d_volume_v1"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels for each day
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1_1d = pivot_1d + (range_1d * 1.0 / 12.0)
    r2_1d = pivot_1d + (range_1d * 1.1 / 12.0)
    r3_1d = pivot_1d + (range_1d * 1.5 / 12.0)
    r4_1d = pivot_1d + (range_1d * 1.6 / 12.0)
    s1_1d = pivot_1d - (range_1d * 1.0 / 12.0)
    s2_1d = pivot_1d - (range_1d * 1.1 / 12.0)
    s3_1d = pivot_1d - (range_1d * 1.5 / 12.0)
    s4_1d = pivot_1d - (range_1d * 1.6 / 12.0)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 day for previous day's levels)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (take profit) or breaks below S3 (stop)
            if close[i] >= r3_1d_aligned[i] or close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches S3 (take profit) or breaks above R3 (stop)
            if close[i] <= s3_1d_aligned[i] or close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Fade at S3/S4: long when price touches S3/S4 with volume
                if close[i] <= s3_1d_aligned[i] or close[i] <= s4_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Fade at R3/R4: short when price touches R3/R4 with volume
                elif close[i] >= r3_1d_aligned[i] or close[i] >= r4_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
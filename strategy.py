#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot from 1D + Volume Confirmation
# Hypothesis: Fade at R3/S3 levels with volume confirmation works in both bull and bear markets.
# Uses mean reversion at extreme daily pivot levels, avoiding false breakouts.
# Target: 15-35 trades/year to minimize fee drag.
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
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 2.0)
    s3 = pivot - (range_1d * 1.1 / 2.0)
    r4 = pivot + (range_1d * 1.1)
    s4 = pivot - (range_1d * 1.1)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=5, min_periods=5).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(5, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 5-day average daily volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 or volume fails
            if close[i] <= s3_aligned[i] or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches R3 or volume fails
            if close[i] >= r3_aligned[i] or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price crosses below S3 with volume (fade the extreme)
            if close[i] < s3_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses above R3 with volume (fade the extreme)
            elif close[i] > r3_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals
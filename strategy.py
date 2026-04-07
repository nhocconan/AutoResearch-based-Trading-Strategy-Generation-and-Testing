#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot from 1d + volume confirmation
# Hypothesis: Fade at R3/S3 levels with strong volume confirmation captures mean reversion
# during ranging markets while avoiding false breakouts. Works in both bull/bear by fading
# extremes rather than following trends. Target: 50-150 total trades over 4 years.
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's OHLC for Camarilla
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (using previous day's range)
    range_prev = prev_high - prev_low
    camarilla_h5 = prev_close + 1.1 * range_prev * 1.1 / 2  # R4 equivalent
    camarilla_h4 = prev_close + 1.1 * range_prev * 1.1 / 4  # R3
    camarilla_h3 = prev_close + 1.1 * range_prev * 1.1 / 6  # R2
    camarilla_l3 = prev_close - 1.1 * range_prev * 1.1 / 6  # S2
    camarilla_l4 = prev_close - 1.1 * range_prev * 1.1 / 4  # S3
    camarilla_l5 = prev_close - 1.1 * range_prev * 1.1 / 2  # S4
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 day for no look-ahead)
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Calculate 6-period volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(6, n):
        # Skip if required data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 6-period average
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 level or volume drops
            if close[i] <= l4_aligned[i] or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches R3 level or volume drops
            if close[i] >= h4_aligned[i] or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price touches S3 level with volume confirmation
            if close[i] <= l4_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R3 level with volume confirmation
            elif close[i] >= h4_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h 12h Camarilla Pivot with Volume Confirmation
# Hypothesis: Price reacting at 12h Camarilla pivot levels (S3/R3 for reversal, S4/R4 for breakout)
# with volume confirmation works in both bull and bear markets. Target: 20-40 trades/year.

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels from previous bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point and range
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_hl = high_12h - low_12h
    
    # Camarilla levels
    r4 = pivot + (range_hl * 1.1 / 2)
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Shift by 1 to use previous 12h bar's data (avoid look-ahead)
    pivot = np.roll(pivot, 1)
    r4 = np.roll(r4, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    
    # Handle first element
    if len(pivot) > 1:
        pivot[0] = pivot[1]
        r4[0] = r4[1]
        r3[0] = r3[1]
        s3[0] = s3[1]
        s4[0] = s4[1]
    else:
        pivot[0] = 0
        r4[0] = 0
        r3[0] = 0
        s3[0] = 0
        s4[0] = 0
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: reversal at S3 or breakout failure at S4
            if (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) or \
               (low[i] >= s4_aligned[i] and close[i] > s4_aligned[i] * 0.995):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: reversal at R3 or breakout failure at R4
            if (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) or \
               (high[i] <= r4_aligned[i] and close[i] < r4_aligned[i] * 1.005):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: reversal at S3 or breakdown below S4 with volume
            if ((low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) or \
                (low[i] < s4_aligned[i] and close[i] < s4_aligned[i])) and \
               vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: reversal at R3 or breakout above R4 with volume
            elif ((high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) or \
                  (high[i] > r4_aligned[i] and close[i] > r4_aligned[i])) and \
                 vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
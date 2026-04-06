#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal with 1d Volume Confirmation v2
Hypothesis: Camarilla pivot levels from daily timeframe provide statistically significant
support/resistance levels. Fade at R3/S3 levels with volume confirmation offers high-probability
mean reversion entries. Works in both bull/bear markets as it fades extremes rather than
following trends. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_reversal_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: range = high - low
    # Resistance levels: R3 = close + (high - low) * 1.1/2, R4 = close + (high - low) * 1.1
    # Support levels: S3 = close - (high - low) * 1.1/2, S4 = close - (high - low) * 1.1
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Handle first value
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    range_1d = prev_high - prev_low
    
    # Calculate levels
    r3 = prev_close + range_1d * 1.1 / 2
    r4 = prev_close + range_1d * 1.1
    s3 = prev_close - range_1d * 1.1 / 2
    s4 = prev_close - range_1d * 1.1
    
    # Align to 6h timeframe (shift by 1 for completed candles)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter - 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For volume MA
    
    for i in range(start, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: mean reversion complete or volatility expansion
        if position == 1:  # long position (entered at S3/S4)
            # Exit: price reaches midpoint (mean reversion complete) OR volatility expansion
            midpoint = (s3_aligned[i] + s4_aligned[i]) / 2
            if close[i] >= midpoint or volume[i] > vol_ma[i] * 3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position (entered at R3/R4)
            # Exit: price reaches midpoint (mean reversion complete) OR volatility expansion
            midpoint = (r3_aligned[i] + r4_aligned[i]) / 2
            if close[i] <= midpoint or volume[i] > vol_ma[i] * 3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: fade extreme levels with volume confirmation
            # Long: price touches S3/S4 with volume confirmation
            if ((abs(close[i] - s3_aligned[i]) < 0.001 * close[i] or 
                 abs(close[i] - s4_aligned[i]) < 0.001 * close[i]) and
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price touches R3/R4 with volume confirmation
            elif ((abs(close[i] - r3_aligned[i]) < 0.001 * close[i] or 
                   abs(close[i] - r4_aligned[i]) < 0.001 * close[i]) and
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal with 1d Volume Confirmation
# Hypothesis: Camarilla levels from daily timeframe provide strong support/resistance. 
# Fade at R3/S3 levels with volume confirmation for mean reversion in ranging markets.
# Works in both bull and bear as price respects these levels during consolidation phases.
# Target: 15-25 trades/year to minimize fee drag.
name = "6h_camarilla_1d_volume_reversal_v1"
timeframe = "6h"
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
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d volume moving average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Camarilla pivots from previous day
    # Using typical price: (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_values = typical_price.values
    
    # Calculate pivots: R4, R3, S3, S4
    # Pivot = typical price of previous day
    pivot = np.roll(typical_price_values, 1)  # Previous day's typical price
    # Range = high - low of previous day
    range_1d = df_1d['high'].values - df_1d['low'].values
    range_1d = np.roll(range_1d, 1)  # Previous day's range
    
    # Camarilla levels
    r3 = pivot + 1.1 * range_1d / 2
    s3 = pivot - 1.1 * range_1d / 2
    r4 = pivot + 1.1 * range_1d
    s4 = pivot - 1.1 * range_1d
    
    # Align pivots to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1d average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 level (support) or closes below pivot
            if close[i] <= s3_aligned[i] or close[i] < pivot[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches R3 level (resistance) or closes above pivot
            if close[i] >= r3_aligned[i] or close[i] > pivot[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price touches S3 level with volume confirmation
            if close[i] <= s3_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R3 level with volume confirmation
            elif close[i] >= r3_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals
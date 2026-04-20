#!/usr/bin/env python3
"""
6h_RangeBreakout_1d_Pivot_SR_With_Volume_Confirmation
Hypothesis: Trade breakouts of 1d pivot-based support/resistance (S1/R1) on 6h timeframe with volume confirmation.
Long when price breaks above S1 (acting as support) with volume > 1.5x 20-period average.
Short when price breaks below R1 (acting as resistance) with volume > 1.5x 20-period average.
Exit when price returns to the pivot point (mean reversion within the daily range).
Works in bull/bear: Uses mean-reversion logic within established daily ranges, effective in ranging markets.
"""

name = "6h_RangeBreakout_1d_Pivot_SR_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Handle first day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    S1 = 2 * pivot - prev_high  # Support 1
    R1 = 2 * pivot - prev_low   # Resistance 1
    
    # Align to 6h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Ensure volume MA and pivot data are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks below S1 (support) with volume spike - expect bounce back up
            if close[i] < S1_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks above R1 (resistance) with volume spike - expect rejection down
            elif close[i] > R1_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot (mean reversion)
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot (mean reversion)
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
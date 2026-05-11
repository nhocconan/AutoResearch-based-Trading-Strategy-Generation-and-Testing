#!/usr/bin/env python3
# 12h Camarilla Pivot Reversal Strategy
# Uses daily Camarilla pivot levels (S3/R3) for mean-reversion entries
# Only trades when price touches these levels with volume confirmation
# Works in both bull/bear markets by fading extremes at key support/resistance
# Target: 15-25 trades/year with tight entry conditions

name = "12h_Camarilla_Pivot_Reversal_1d"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = close + ((high - low) * 1.5)
    # R3 = close + ((high - low) * 1.25)
    # R2 = close + ((high - low) * 1.166)
    # R1 = close + ((high - low) * 1.083)
    # S1 = close - ((high - low) * 1.083)
    # S2 = close - ((high - low) * 1.166)
    # S3 = close - ((high - low) * 1.25)
    # S4 = close - ((high - low) * 1.5)
    
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Calculate pivot levels
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.25
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.25
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.5
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.5
    
    # Align to 12h timeframe (wait for daily bar to close)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > 1.5 * vol_ma24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need volume MA data
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price touches S3 level with volume confirmation
            if low[i] <= s3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price touches R3 level with volume confirmation
            elif high[i] >= r3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches S4 level or mean reversion to midpoint
            midpoint = (s3_aligned[i] + r3_aligned[i]) / 2
            if high[i] >= s4_aligned[i] or close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches R4 level or mean reversion to midpoint
            midpoint = (s3_aligned[i] + r3_aligned[i]) / 2
            if low[i] <= r4_aligned[i] or close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
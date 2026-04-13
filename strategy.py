#!/usr/bin/env python3
"""
6h_1d_Weekly_Pivot_Breakout_With_Volume
Hypothesis: Weekly pivot levels provide stronger support/resistance than daily in ranging markets.
Breakouts above weekly R3 or below weekly S3 on 6h chart with volume expansion capture institutional moves.
Works in both bull and bear markets by trading breakouts regardless of direction.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot formulas (classic)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    R1 = 2 * pivot - low_1w
    S1 = 2 * pivot - high_1w
    R2 = pivot + (high_1w - low_1w)
    S2 = pivot - (high_1w - low_1w)
    R3 = high_1w + 2 * (pivot - low_1w)
    S3 = low_1w - 2 * (high_1w - pivot)
    
    # Align weekly levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above weekly R3 with volume expansion
        long_breakout = close[i] > R3_aligned[i] and volume_expansion[i]
        
        # Short breakdown: price breaks below weekly S3 with volume expansion
        short_breakout = close[i] < S3_aligned[i] and volume_expansion[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_Weekly_Pivot_Breakout_With_Volume"
timeframe = "6h"
leverage = 1.0
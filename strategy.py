#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout
Hypothesis: Combines weekly Camarilla pivot levels with daily price action to trade breakouts.
In ranging markets, price tends to revert to the mean (pivot); in trending markets, breaks of
H3/L3 levels indicate momentum. Uses volume confirmation to filter false breaks.
Works in both bull and bear markets by adapting to regime via price action relative to pivot.
Target: 10-25 trades/year on 1d (40-100 total over 4 years).
"""

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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week
    high_prev = df_1w['high'].shift(1).values  # Previous week high
    low_prev = df_1w['low'].shift(1).values    # Previous week low
    close_prev = df_1w['close'].shift(1).values # Previous week close
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    # Levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We use H3 and L3 for breakouts
    camarilla_h3 = close_prev + (range_prev * 1.1 / 6)
    camarilla_l3 = close_prev - (range_prev * 1.1 / 6)
    
    # Align weekly levels to daily timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Get daily data for entry and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume_1d > (vol_ma_20 * 1.5)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion.values)
    
    # Pivot (mean) for mean reversion in range
    camarilla_pivot = (high_prev + low_prev + close_prev) / 3
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(camarilla_h3_aligned[i]) or \
           np.isnan(camarilla_l3_aligned[i]) or \
           np.isnan(camarilla_pivot_aligned[i]) or \
           np.isnan(volume_expansion_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: price above H3 with volume expansion OR below pivot with mean reversion
        long_breakout = close[i] > camarilla_h3_aligned[i] and volume_expansion_aligned[i]
        long_reversion = close[i] < camarilla_pivot_aligned[i] and \
                        close[i] > camarilla_l3_aligned[i] and \
                        volume_expansion_aligned[i]
        
        # Short conditions: price below L3 with volume expansion OR above pivot with mean reversion
        short_breakout = close[i] < camarilla_l3_aligned[i] and volume_expansion_aligned[i]
        short_reversion = close[i] > camarilla_pivot_aligned[i] and \
                         close[i] < camarilla_h3_aligned[i] and \
                         volume_expansion_aligned[i]
        
        # Entry logic
        if long_breakout or long_reversion:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        elif short_breakout or short_reversion:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        else:
            # Hold or flat
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_Camarilla_Pivot_Breakout"
timeframe = "1d"
leverage = 1.0
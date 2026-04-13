#!/usr/bin/env python3
"""
4h_1d_camarilla_pivot_volume
Hypothesis: Uses 1-day Camarilla pivot levels with volume confirmation to capture reversals at key support/resistance.
Works in bull markets (bounce from support) and bear markets (rejection at resistance).
Targets 20-30 trades/year to minimize fee drag. Long and short positions for symmetry.
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
    
    # Get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Using formula: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low), L3 = Close - 1.125*(High-Low)
    # We'll use H3/L3 as entry levels
    hl_range = high_1d - low_1d
    H3 = close_1d + 1.125 * hl_range
    L3 = close_1d - 1.125 * hl_range
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_expansion = df_1d['volume'].values > (vol_ma_20 * 1.5)
    
    # Align all signals to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price touches H3/L3 with volume expansion
        long_entry = (low[i] <= L3_aligned[i]) and volume_expansion_aligned[i] > 0.5
        short_entry = (high[i] >= H3_aligned[i]) and volume_expansion_aligned[i] > 0.5
        
        # Exit conditions: return to previous day's close
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        exit_long = position == 1 and close[i] >= prev_close_aligned[i]
        exit_short = position == -1 and close[i] <= prev_close_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_pivot_volume"
timeframe = "4h"
leverage = 1.0
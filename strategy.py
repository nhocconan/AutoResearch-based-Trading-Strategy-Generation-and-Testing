#!/usr/bin/env python3
"""
4h_1d_1w_Pivot_Confluence_Scalp
Hypothesis: Confluence of 1d and 1w Camarilla pivot levels creates strong support/resistance.
Breakouts above weekly R4 or below weekly S4 with volume expansion capture institutional moves.
Works in bull markets (breakouts up) and bear markets (breakdowns down) due to symmetry.
Target: 25-35 trades/year by requiring multi-timeframe confluence and volume filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    range_val = high - low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    C = close
    H = high
    L = low
    R1 = C + ((H - L) * 1.0833)
    R2 = C + ((H - L) * 1.1666)
    R3 = C + ((H - L) * 1.2500)
    R4 = C + ((H - L) * 1.5000)
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for confluence
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot levels
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    R1_1w, R2_1w, R3_1w, R4_1w, S1_1w, S2_1w, S3_1w, S4_1w = calculate_camarilla(high_1w, low_1w, close_1w)
    
    # Align to 4h timeframe
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.28
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or 
            np.isnan(R4_1w_aligned[i]) or np.isnan(S4_1w_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above BOTH 1d R4 AND 1w R4 with volume expansion
        long_condition = (close[i] > R4_1d_aligned[i]) and (close[i] > R4_1w_aligned[i]) and volume_expansion[i]
        
        # Short: breakdown below BOTH 1d S4 AND 1w S4 with volume expansion
        short_condition = (close[i] < S4_1d_aligned[i]) and (close[i] < S4_1w_aligned[i]) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_1w_Pivot_Confluence_Scalp"
timeframe = "4h"
leverage = 1.0
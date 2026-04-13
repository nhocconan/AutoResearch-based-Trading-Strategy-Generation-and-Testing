#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v1
Hypothesis: Weekly and daily Camarilla pivot levels (S4/R4) act as major support/resistance.
Breakouts above weekly R4 or below weekly S4 on 12h chart with volume expansion capture
institutional moves. Uses daily S3/R3 for additional confluence. Works in both bull and bear
markets by trading breakouts regardless of direction. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly and daily data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (S4/R4)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    close_prev_1w = np.roll(close_1w, 1)
    close_prev_1w[0] = close_1w[0]
    range_1w = high_1w - low_1w
    
    # Weekly R4 and S4 (strongest levels)
    R4_1w = close_prev_1w + (range_1w * 1.5000 / 2)  # R4 = close + 1.5*(range)/2
    S4_1w = close_prev_1w - (range_1w * 1.5000 / 2)  # S4 = close - 1.5*(range)/2
    
    # Calculate daily Camarilla levels (S3/R3 for confluence)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_prev_1d = np.roll(close_1d, 1)
    close_prev_1d[0] = close_1d[0]
    range_1d = high_1d - low_1d
    
    # Daily R3 and S3
    R3_1d = close_prev_1d + (range_1d * 1.2500 / 4)
    S3_1d = close_prev_1d - (range_1d * 1.2500 / 4)
    
    # Align levels to 12h timeframe
    R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # Volume confirmation: current volume > 1.8x 30-period average (stricter for 12h)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    volume_expansion = volume > (vol_ma_30 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_1w_aligned[i]) or np.isnan(S4_1w_aligned[i]) or 
            np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above weekly R4 with volume expansion AND above daily R3
        long_breakout = (close[i] > R4_1w_aligned[i] and 
                        close[i] > R3_1d_aligned[i] and 
                        volume_expansion[i])
        
        # Short breakdown: price breaks below weekly S4 with volume expansion AND below daily S3
        short_breakout = (close[i] < S4_1w_aligned[i] and 
                         close[i] < S3_1d_aligned[i] and 
                         volume_expansion[i])
        
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

name = "12h_1w_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0
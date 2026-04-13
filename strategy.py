#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Daily Camarilla pivot levels provide high-probability support/resistance.
Breakouts above H4 or below L4 with volume expansion indicate strong institutional moves.
12h timeframe reduces noise and overtrading while capturing multi-day trends.
Works in bull markets (breakouts continue) and bear markets (fades at resistance) via price action.
Volume confirmation filters false breakouts. Targets 15-30 trades/year.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # Formula: Range = high - low
    # H4 = close + (Range * 1.1/2)
    # L4 = close - (Range * 1.1/2)
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)
    camarilla_h3 = close_1d + (range_1d * 1.1/4)
    camarilla_l3 = close_1d - (range_1d * 1.1/4)
    camarilla_h2 = close_1d + (range_1d * 1.1/6)
    camarilla_l2 = close_1d - (range_1d * 1.1/6)
    camarilla_h1 = close_1d + (range_1d * 1.1/12)
    camarilla_l1 = close_1d - (range_1d * 1.1/12)
    
    # Align daily Camarilla levels to 12h
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    volume_expansion = volume > (vol_ma_30 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Breakout above H4 with volume expansion
        # 2. Optional: above H3 for stronger signal (not required)
        breakout_long = (close[i] > h4_1d_aligned[i]) and volume_expansion[i]
        
        # Short conditions:
        # 1. Breakdown below L4 with volume expansion
        # 2. Optional: below L3 for stronger signal (not required)
        breakdown_short = (close[i] < l4_1d_aligned[i]) and volume_expansion[i]
        
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakdown_short and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0
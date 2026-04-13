#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_Volume
Hypothesis: Combines Camarilla pivot levels from 12h with breakout confirmation on 4h.
Uses Camarilla levels (H4/L4) as key support/resistance. Enters on break of these levels with volume confirmation.
Works in both bull and bear markets by trading breakouts from key pivot levels.
Target: 20-50 trades/year on 4h (80-200 total over 4 years).
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # H4 = Close + 1.1 * (High - Low) / 2
    # L4 = Close - 1.1 * (High - Low) / 2
    camarilla_h4 = close_12h + 1.1 * (high_12h - low_12h) / 2
    camarilla_l4 = close_12h - 1.1 * (high_12h - low_12h) / 2
    
    # Get 4h data for breakout confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    volume_expansion_4h = volume_4h > (vol_ma_20_4h * 1.5)
    
    # Align all signals to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    volume_expansion_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_expansion_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    # Track pivot levels for entry
    h4_level = np.zeros(n)
    l4_level = np.zeros(n)
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(camarilla_h4_aligned[i]) or \
           np.isnan(camarilla_l4_aligned[i]) or \
           np.isnan(volume_expansion_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update pivot levels when new 12h bar completes
        h4_level[i] = camarilla_h4_aligned[i]
        l4_level[i] = camarilla_l4_aligned[i]
        
        # Entry conditions: price breaks Camarilla levels with volume expansion
        if volume_expansion_4h_aligned[i]:
            # Long entry: price breaks above H4 level
            if close[i] > h4_level[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Short entry: price breaks below L4 level
            elif close[i] < l4_level[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Hold current position
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # No volume expansion - hold or flat
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_Camarilla_Pivot_Breakout_Volume"
timeframe = "4h"
leverage = 1.0
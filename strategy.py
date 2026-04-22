#!/usr/bin/env python3
"""
Hypothesis: 12-hour Camarilla pivot reversal with 1-day volume filter.
Long when price breaks above H3 and 1-day volume > 1.5x average.
Short when price breaks below L3 and 1-day volume > 1.5x average.
Exit when price crosses H4/L4 or volume condition fails.
Camarilla levels provide institutional support/resistance; volume filter ensures participation.
Works in both bull and bear markets by capturing reversals at key levels with volume confirmation.
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
    
    # Load 1-day data for Camarilla and volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H4, H3, L3, L4
    range_1d = high_1d - low_1d
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]
    
    H3 = close_prev + 1.1 * range_1d / 6
    L3 = close_prev - 1.1 * range_1d / 6
    H4 = close_prev + 1.1 * range_1d / 2
    L4 = close_prev - 1.1 * range_1d / 2
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume filter: 1-day volume > 1.5x 50-day average
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    vol_condition = volume_1d > (1.5 * avg_vol_1d)
    vol_aligned = align_htf_to_ltf(prices, df_1d, vol_condition)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above H3 with volume confirmation
            if close[i] > H3_aligned[i] and vol_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 with volume confirmation
            elif close[i] < L3_aligned[i] and vol_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses H4 or volume fails
                if close[i] > H4_aligned[i] or not vol_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses L4 or volume fails
                if close[i] < L4_aligned[i] or not vol_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_H3L3_Breakout_1dVolumeFilter"
timeframe = "12h"
leverage = 1.0
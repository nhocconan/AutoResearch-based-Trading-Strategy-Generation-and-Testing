#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v5
Hypothesis: On 12-hour timeframe, trade reversals at daily Camarilla pivot levels (H3/L3, H4/L4) with volume confirmation.
Long when price touches L3/L4 with volume > 1.5x average; short when touches H3/H4 with volume > 1.5x average.
Exit on opposite touch or when price moves beyond H5/L5. Works in ranging markets (2025-2026) and captures mean reversion.
Designed for 15-30 trades/year to minimize fee drag while capturing reversals in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v5"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    H4 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 2
    H3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    L3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    L4 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 2
    H5 = close_1d + 1.1 * (high_1d - low_1d) * 1.1
    L5 = close_1d - 1.1 * (high_1d - low_1d) * 1.1
    
    # Align to 12h timeframe
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    H5_12h = align_htf_to_ltf(prices, df_1d, H5)
    L5_12h = align_htf_to_ltf(prices, df_1d, L5)
    
    # Volume filter: 1.5x average volume
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if pivot data not available
        if np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]):
            signals[i] = 0.0
            continue
            
        # Long conditions: touch L3 or L4 with volume confirmation
        long_signal = (
            (low[i] <= L3_12h[i] * 1.001 or low[i] <= L4_12h[i] * 1.001) and
            vol_filter[i]
        )
        
        # Short conditions: touch H3 or H4 with volume confirmation
        short_signal = (
            (high[i] >= H3_12h[i] * 0.999 or high[i] >= H4_12h[i] * 0.999) and
            vol_filter[i]
        )
        
        # Exit conditions: opposite touch or beyond H5/L5
        exit_long = (
            high[i] >= H3_12h[i] * 0.999 or  # Touch H3 on long
            high[i] >= H5_12h[i] * 0.999     # Beyond H5
        )
        
        exit_short = (
            low[i] <= L3_12h[i] * 1.001 or   # Touch L3 on short
            low[i] <= L5_12h[i] * 1.001      # Beyond L5
        )
        
        if position == 1:  # Long position
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if long_signal:
                position = 1
                signals[i] = 0.25
            elif short_signal:
                position = -1
                signals[i] = -0.25
    
    return signals
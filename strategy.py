#!/usr/bin/env python3
"""
12h_1w_camarilla_breakout_volume
Uses 1w Camarilla pivot levels on 12h timeframe with volume confirmation.
Long when price breaks above H4 resistance, short when breaks below L4 support.
Exit when price returns to H3/L3 levels.
Volume filter: current volume > 1.8x 24-period average to avoid false breakouts.
Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
Works in both trending and ranging markets by combining institutional levels with volume confirmation.
"""

name = "12h_1w_camarilla_breakout_volume"
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
    
    # Get 1w data for Camarilla calculation (weekly pivot levels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels (based on previous week)
    # Typical pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = Close + 1.1 * Range * 1.1
    # L4 = Close - 1.1 * Range * 1.1
    # H3 = Close + 1.1 * Range * 0.5
    # L3 = Close - 1.1 * Range * 0.5
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    h4 = close_1w + 1.1 * range_1w * 1.1
    l4 = close_1w - 1.1 * range_1w * 1.1
    h3 = close_1w + 1.1 * range_1w * 0.5
    l3 = close_1w - 1.1 * range_1w * 0.5
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Volume confirmation: volume > 1.8x 24-period average (2 days of 12h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for calculations
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above H4 resistance with volume confirmation
        if close[i] > h4_aligned[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below L4 support with volume confirmation
        elif close[i] < l4_aligned[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: return to H3/L3 levels (profit taking)
        elif position == 1 and close[i] <= h3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= l3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
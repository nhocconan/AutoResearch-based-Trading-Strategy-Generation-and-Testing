#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot levels and volume confirmation.
# Long: Price crosses above weekly R4 level + volume > 2x average volume (20-period).
# Short: Price crosses below weekly S4 level + volume > 2x average volume.
# Uses weekly pivot levels for major support/resistance structure, 6h for execution.
# Weekly pivots provide strong institutional levels that work in both bull and bear markets.
# Volume confirmation filters out false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels (using previous week's data)
    pivot = np.full(len(close_1w), np.nan)
    r4 = np.full(len(close_1w), np.nan)
    s4 = np.full(len(close_1w), np.nan)
    for i in range(1, len(close_1w)):
        # Previous week's OHLC
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        # Standard pivot point
        pp = (ph + pl + pc) / 3.0
        
        # Weekly R4 and S4 levels
        r4[i] = pc + 3 * (ph - pl)  # R4 = Close + 3*(High-Low)
        s4[i] = pc - 3 * (ph - pl)  # S4 = Close - 3*(High-Low)
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align weekly pivot levels to 6h
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r4_level = r4_aligned[i]
        s4_level = s4_aligned[i]
        
        # Volume confirmation: current volume > 2x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: price crosses above R4 + volume confirmation
            if (price > r4_level and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price crosses below S4 + volume confirmation
            elif (price < s4_level and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below S4 (opposite level)
            if price < s4_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above R4 (opposite level)
            if price > r4_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_R4S4_Volume"
timeframe = "6h"
leverage = 1.0
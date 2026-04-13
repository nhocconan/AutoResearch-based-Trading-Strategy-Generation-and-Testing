#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot levels and volume confirmation.
# Long: Price crosses above weekly R4 level + volume > 2x average volume (24-period).
# Short: Price crosses below weekly S4 level + volume > 2x average volume.
# Uses weekly pivot levels for major support/resistance structure, 6h for execution with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
# Weekly pivots provide strong institutional levels that work in both bull and bear markets.

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
    
    # Calculate weekly pivot points and levels
    pivot = np.full(len(close_1w), np.nan)
    r1 = np.full(len(close_1w), np.nan)
    r2 = np.full(len(close_1w), np.nan)
    r3 = np.full(len(close_1w), np.nan)
    r4 = np.full(len(close_1w), np.nan)
    s1 = np.full(len(close_1w), np.nan)
    s2 = np.full(len(close_1w), np.nan)
    s3 = np.full(len(close_1w), np.nan)
    s4 = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        # Previous week's OHLC
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        # Pivot point
        pivot[i] = (ph + pl + pc) / 3.0
        
        # Support and resistance levels
        r1[i] = 2 * pivot[i] - pl
        s1[i] = 2 * pivot[i] - ph
        r2[i] = pivot[i] + (ph - pl)
        s2[i] = pivot[i] - (ph - pl)
        r3[i] = ph + 2 * (pivot[i] - pl)
        s3[i] = pl - 2 * (ph - pivot[i])
        r4[i] = r3[i] + (ph - pl)
        s4[i] = s3[i] - (ph - pl)
    
    # Average volume (24-period = 4 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    # Align weekly pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        
        # Volume confirmation: current volume > 2x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: price crosses above R4 + volume confirmation
            if (price > r4 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price crosses below S4 + volume confirmation
            elif (price < s4 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below pivot (mean reversion to mean)
            if price < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above pivot (mean reversion to mean)
            if price > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_R4S4_Volume"
timeframe = "6h"
leverage = 1.0
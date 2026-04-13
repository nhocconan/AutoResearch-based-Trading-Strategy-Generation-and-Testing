#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot levels and volume confirmation.
# Long: Price crosses above weekly R1 level + volume > 1.8x average volume (20-period).
# Short: Price crosses below weekly S1 level + volume > 1.8x average volume.
# Uses weekly pivot levels for support/resistance structure, 6h for execution with volume confirmation.
# Pivot calculation: P = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H.
# Time filter: 00-23 UTC (all hours) to maximize opportunities while maintaining discipline.
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
    pivot_p = np.full(len(close_1w), np.nan)
    pivot_r1 = np.full(len(close_1w), np.nan)
    pivot_s1 = np.full(len(close_1w), np.nan)
    for i in range(1, len(close_1w)):
        # Previous week's OHLC
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        # Pivot point
        p = (ph + pl + pc) / 3.0
        pivot_p[i] = p
        # R1 and S1
        pivot_r1[i] = 2 * p - pl
        pivot_s1[i] = 2 * p - ph
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align weekly pivot levels to 6h
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1w, pivot_r1)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1w, pivot_s1)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_r1_aligned[i]) or np.isnan(pivot_s1_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r1 = pivot_r1_aligned[i]
        s1 = pivot_s1_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirm = vol > 1.8 * avg_vol
        
        if position == 0:
            # Long: price crosses above R1 + volume confirmation
            if (price > r1 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price crosses below S1 + volume confirmation
            elif (price < s1 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below S1 (opposite level)
            if price < s1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above R1 (opposite level)
            if price > r1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_R1S1_Volume"
timeframe = "6h"
leverage = 1.0
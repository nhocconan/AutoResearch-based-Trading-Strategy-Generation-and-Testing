#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume
Hypothesis: Combines Camarilla pivot levels from 1d with breakout confirmation and volume filter.
In ranging markets, price often reverts to mean near pivot levels; in trending markets,
breakouts from key pivot levels (S3/S4 for longs, R3/R4 for shorts) with volume confirmation
provide high-probability entries. Works in both bull and bear markets by adapting to
price action around key institutional levels. Target: 20-50 trades/year on 4h.
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
    
    # Calculate Camarilla pivot levels for each day
    # R4 = Close + ((High - Low) * 1.5)
    # R3 = Close + ((High - Low) * 1.25)
    # R2 = Close + ((High - Low) * 1.166)
    # R1 = Close + ((High - Low) * 1.083)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.083)
    # S2 = Close - ((High - Low) * 1.166)
    # S3 = Close - ((High - Low) * 1.25)
    # S4 = Close - ((High - Low) * 1.5)
    
    pivot_points = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    r4 = close_1d + (range_hl * 1.5)
    r3 = close_1d + (range_hl * 1.25)
    r2 = close_1d + (range_hl * 1.166)
    r1 = close_1d + (range_hl * 1.083)
    s1 = close_1d - (range_hl * 1.083)
    s2 = close_1d - (range_hl * 1.166)
    s3 = close_1d - (range_hl * 1.25)
    s4 = close_1d - (range_hl * 1.5)
    
    # Align pivot levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_points)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Choppiness regime filter (optional, can be added if needed)
    # For now, rely on pivot levels and volume
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above S3 or S4 with volume expansion
        # Actually, we want breakouts ABOVE resistance for longs, BELOW support for shorts
        # But Camarilla suggests S3/S4 as strong support, R3/R4 as strong resistance
        # Let's interpret as: break above R3/R4 = long, break below S3/S4 = short
        
        long_breakout = (close[i] > r3_aligned[i] or close[i] > r4_aligned[i]) and volume_expansion[i]
        short_breakout = (close[i] < s3_aligned[i] or close[i] < s4_aligned[i]) and volume_expansion[i]
        
        # Exit conditions: return to pivot point or opposite pivot level
        long_exit = close[i] < pp_aligned[i]
        short_exit = close[i] > pp_aligned[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "4h"
leverage = 1.0
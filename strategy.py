#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_R1S1_Breakout_Volume_Confirmation_V1
Hypothesis: Use weekly and daily confluence on 12h timeframe. Enter long when price breaks above daily R1 AND weekly R1 with volume > 2.0x 20-day average volume. Enter short when price breaks below daily S1 AND weekly S1 with volume filter. Exit on opposite breakout. Weekly confluence filters false breaks, volume ensures momentum, reducing trades to target 50-150 over 4 years. Designed to work in both bull (breakouts) and bear (mean reversion at extremes) via strict entry.
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
    
    # === Daily data for Camarilla levels and volume ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Weekly data for confluence filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily Camarilla levels
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = pp_1d + (range_1d * 1.1 / 12.0)
    s1_1d = pp_1d - (range_1d * 1.1 / 12.0)
    
    # Calculate weekly Camarilla levels
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = pp_1w + (range_1w * 1.1 / 12.0)
    s1_1w = pp_1w - (range_1w * 1.1 / 12.0)
    
    # Align daily levels to 12h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Align weekly levels to 12h
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily volume average for confirmation (20-day)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Current daily volume aligned
    vol_1d_current_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers 20-day volume average
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_current_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current daily volume > 2.0x 20-day average
        vol_filter = vol_1d_current_aligned[i] > 2.0 * vol_avg_20_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above BOTH daily R1 and weekly R1 + volume filter
            if (close[i] > r1_1d_aligned[i] and close[i] > r1_1w_aligned[i] and vol_filter):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below BOTH daily S1 and weekly S1 + volume filter
            elif (close[i] < s1_1d_aligned[i] and close[i] < s1_1w_aligned[i] and vol_filter):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal (break opposite level on either timeframe)
        elif position == 1:
            # Exit when price breaks below either daily S1 or weekly S1
            if close[i] < s1_1d_aligned[i] or close[i] < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above either daily R1 or weekly R1
            if close[i] > r1_1d_aligned[i] or close[i] > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_1d_Camarilla_R1S1_Breakout_Volume_Confirmation_V1"
timeframe = "12h"
leverage = 1.0
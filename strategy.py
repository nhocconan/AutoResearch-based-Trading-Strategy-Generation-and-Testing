#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_With_Volume_Filter
Hypothesis: Daily chart strategy using weekly timeframe context. Enter long when price breaks above daily R1 with volume confirmation and weekly uptrend (weekly close > weekly open). Enter short when price breaks below daily S1 with volume confirmation and weekly downtrend. Exit on opposite break. Designed for low trade frequency to work in both bull and bear markets by using weekly trend filter to avoid counter-trend trades.
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
    
    # === Daily data for signal generation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Camarilla pivot levels
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    r1 = pp + (range_hl * 1.1 / 12.0)
    s1 = pp - (range_hl * 1.1 / 12.0)
    
    # Align to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily volume average for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # === Weekly trend filter (1-week close > open for uptrend) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_uptrend = close_1w > open_1w
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period: enough for daily calculations
    warmup = 30  # Covers 20-day volume average
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(weekly_uptrend_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x daily average volume
        vol_filter = vol_1d_current > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above R1 + volume filter + weekly uptrend
            if close[i] > r1_aligned[i] and vol_filter and weekly_uptrend_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below S1 + volume filter + weekly downtrend
            elif close[i] < s1_aligned[i] and vol_filter and weekly_uptrend_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal
        elif position == 1:
            # Exit when price breaks below S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1S1_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0
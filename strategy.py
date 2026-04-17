#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_Volume_Strict
Hypothesis: On 12h timeframe, trade Camarilla pivot R1/S1 breakouts with daily volume confirmation to capture institutional breakouts while avoiding false signals. Designed for low trade frequency (12-37/year) to minimize fee drag and work in both bull and bear markets by using daily structure for trend context and volume filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """
    Calculate Camarilla pivot levels:
    R4 = close + (high - low) * 1.5000
    R3 = close + (high - low) * 1.2500
    R2 = close + (high - low) * 1.1666
    R1 = close + (high - low) * 1.0833
    PP = (high + low + close) / 3
    S1 = close - (high - low) * 1.0833
    S2 = close - (high - low) * 1.1666
    S3 = close - (high - low) * 1.2500
    S4 = close - (high - low) * 1.5000
    """
    range_hl = high - low
    r1 = close + range_hl * 1.0833
    s1 = close - range_hl * 1.0833
    pp = (high + low + close) / 3.0
    return r1, s1, pp

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Data (HTF for Camarilla pivot, volume) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Camarilla levels
    r1_1d, s1_1d, pp_1d = calculate_camarilla_pivot(high_1d, low_1d, close_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Daily average volume (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(pp_1d_aligned[i]) or
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 2.0x daily average volume
        vol_filter = vol_1d_current > 2.0 * vol_avg_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with price above pivot point and volume filter
            if close[i] > r1_1d_aligned[i] and close[i] > pp_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with price below pivot point and volume filter
            elif close[i] < s1_1d_aligned[i] and close[i] < pp_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price breaks below S1 (opposite level)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above R1 (opposite level)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1_S1_Breakout_Volume_Strict"
timeframe = "12h"
leverage = 1.0
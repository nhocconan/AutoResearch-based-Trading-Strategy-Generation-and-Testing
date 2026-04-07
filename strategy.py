#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on daily timeframe
combined with volume confirmation works on 12h timeframe. In ranging markets, fade at R3/S3; in trending markets,
breakout at R4/S4. Volume confirmation reduces false signals. Targets 12-37 trades/year (50-150 over 4 years).
Uses daily data for structure, 12h for execution. Works in both bull and bear markets by adapting to price
action relative to pivots.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align daily levels to 12h timeframe
    r4_12h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 20-period volume average on 12h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 (mean reversion fail) OR 
            # price breaks above R4 and fails to hold (breakout fail)
            if close[i] < s3_12h[i] or (close[i] > r4_12h[i] and close[i] < r3_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (mean reversion fail) OR
            # price breaks below S4 and fails to hold (breakout fail)
            if close[i] > r3_12h[i] or (close[i] < s4_12h[i] and close[i] > s3_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion longs at S3
            if (close[i] <= s3_12h[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Mean reversion shorts at R3
            elif (close[i] >= r3_12h[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
            # Breakout longs at R4
            elif (close[i] >= r4_12h[i] and 
                  vol_confirm):
                position = 1
                signals[i] = 0.25
            # Breakout shorts at S4
            elif (close[i] <= s4_12h[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 4h_camarilla_pivot_1d_volume_v1
# Hypothesis: Camarilla pivot levels from daily timeframe combined with volume confirmation on 4h.
# Long when price closes above Camarilla resistance level (R3) with volume > 1.5x average.
# Short when price closes below Camarilla support level (S3) with volume > 1.5x average.
# Exit on opposite signal or when volume drops below average.
# Uses Camarilla levels derived from daily high/low/close to capture institutional support/resistance.
# Volume filter reduces whipsaw. Target: 50-100 total trades over 4 years (~12-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily data
    # Using previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R4 = close_1d + range_1d * 1.1 / 2
    R3 = close_1d + range_1d * 1.1 / 4
    R2 = close_1d + range_1d * 1.1 / 6
    R1 = close_1d + range_1d * 1.1 / 12
    S1 = close_1d - range_1d * 1.1 / 12
    S2 = close_1d - range_1d * 1.1 / 6
    S3 = close_1d - range_1d * 1.1 / 4
    S4 = close_1d - range_1d * 1.1 / 2
    
    # We'll use R3 and S3 as entry levels
    camarilla_R3 = R3
    camarilla_S3 = S3
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 or volume drops below average
            if close[i] < camarilla_S3_aligned[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 or volume drops below average
            if close[i] > camarilla_R3_aligned[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Price action entries at Camarilla levels
            if close[i] > camarilla_R3_aligned[i] and volume_ok:
                position = 1
                signals[i] = 0.25
            elif close[i] < camarilla_S3_aligned[i] and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals
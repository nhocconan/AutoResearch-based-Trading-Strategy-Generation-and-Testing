#!/usr/bin/env python3
# 6h_camilla_pivot_breakout_volume_v1
# Hypothesis: Uses daily Camarilla pivot levels with breakout/continuation logic and volume confirmation.
# Long when: Price breaks above R4 with volume > 1.5x average.
# Short when: Price breaks below S4 with volume > 1.5x average.
# Exit when: Price returns to R3/S3 or volume drops below average.
# Uses daily pivots for structure and volume for confirmation to reduce false breakouts.
# Target: 12-37 trades/year on 6h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camilla_pivot_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not np.isnan(high_1d[i]) and not np.isnan(low_1d[i]) and not np.isnan(close_1d[i]):
            diff = high_1d[i] - low_1d[i]
            camarilla_r4[i] = close_1d[i] + 1.5 * diff
            camarilla_r3[i] = close_1d[i] + 1.1 * diff
            camarilla_s3[i] = close_1d[i] - 1.1 * diff
            camarilla_s4[i] = close_1d[i] - 1.5 * diff
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = vol_ma_period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to R3 or volume drops below average
            if close[i] <= camarilla_r3_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to S3 or volume drops below average
            if close[i] >= camarilla_s3_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above R4 with volume surge
            if close[i] > camarilla_r4_aligned[i] and vol_surge[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S4 with volume surge
            elif close[i] < camarilla_s4_aligned[i] and vol_surge[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
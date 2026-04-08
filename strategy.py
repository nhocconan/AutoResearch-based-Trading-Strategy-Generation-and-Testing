#!/usr/bin/env python3
# 4h_camarilla_volume_confluence_v1
# Hypothesis: Uses 1-day Camarilla pivot levels with volume confirmation for mean reversion.
# Long when price touches S3 with volume surge, short when price touches R3 with volume surge.
# Exit when price returns to mean (Pivot) or volume drops.
# Works in both bull/bear markets by fading extremes at institutional pivot levels.
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_volume_confluence_v1"
timeframe = "4h"
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
    
    # Get 1-day data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    camarilla_high = np.full(len(close_1d), np.nan)
    camarilla_low = np.full(len(close_1d), np.nan)
    camarilla_pivot = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not np.isnan(high_1d[i]) and not np.isnan(low_1d[i]) and not np.isnan(close_1d[i]):
            camarilla_pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
            camarilla_r3[i] = camarilla_pivot[i] + 1.1 * (high_1d[i] - low_1d[i]) / 6
            camarilla_s3[i] = camarilla_pivot[i] - 1.1 * (high_1d[i] - low_1d[i]) / 6
            camarilla_high[i] = high_1d[i]
            camarilla_low[i] = low_1d[i]
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot or volume drops
            if close[i] >= camarilla_pivot_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot or volume drops
            if close[i] <= camarilla_pivot_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price at or below S3 with volume surge
            if (close[i] <= camarilla_s3_aligned[i] and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price at or above R3 with volume surge
            elif (close[i] >= camarilla_r3_aligned[i] and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
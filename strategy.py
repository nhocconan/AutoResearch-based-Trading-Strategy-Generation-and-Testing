#!/usr/bin/env python3
"""
6h Camarilla Pivot from 1-day with Volume Confirmation
Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) from daily chart act as strong support/resistance.
At 6h timeframe, we fade touches of R3/S3 (mean reversion) and break through R4/S4 (momentum).
Volume confirmation filters false signals. Works in ranging markets (fade) and trending markets (breakout).
Target: 15-25 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_volume_v1"
timeframe = "6h"
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
    
    # 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1-day bar
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    r4_1d = close_1d + 1.5 * (high_1d - low_1d)
    r3_1d = close_1d + 1.0 * (high_1d - low_1d)
    s3_1d = close_1d - 1.0 * (high_1d - low_1d)
    s4_1d = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align to 6h timeframe (already shifted by 1 day in align function)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (support) or breaks below S4 with volume
            if close[i] <= s3_6h[i] or (close[i] < s4_6h[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R3 (resistance) or breaks above R4 with volume
            if close[i] >= r3_6h[i] or (close[i] > r4_6h[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at R3/S3: price touches level and reverses
            # Long: touch S3 with rejection (low touches S3 but close above)
            if low[i] <= s3_6h[i] and close[i] > s3_6h[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: touch R3 with rejection (high touches R3 but close below)
            elif high[i] >= r3_6h[i] and close[i] < r3_6h[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
            # Breakout through R4/S4 with volume
            # Long: break above R4
            elif high[i] > r4_6h[i] and close[i] > r4_6h[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: break below S4
            elif low[i] < s4_6h[i] and close[i] < s4_6h[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
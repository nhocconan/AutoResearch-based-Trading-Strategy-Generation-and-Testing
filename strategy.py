#!/usr/bin/env python3
# 6m_1w_1d_pivot_volume_breakout_v1
# Hypothesis: Fade at weekly pivot R3/S3, breakout continuation at weekly R4/S4.
# Uses weekly Camarilla pivot levels with volume confirmation for breakouts.
# In ranging markets: fade extreme weekly pivots (R3/S3) with volume spike.
# In trending markets: breakout beyond weekly R4/S4 with volume surge.
# Weekly pivot provides structure, volume confirms conviction.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6m_1w_1d_pivot_volume_breakout_v1"
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
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous weekly bar
    # H, L, C from previous weekly candle
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Camarilla formulas
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    range_h_l = prev_week_high - prev_week_low
    
    # Resistance levels
    r3 = pivot + (range_h_l * 1.1 / 2)
    r4 = pivot + (range_h_l * 1.1)
    
    # Support levels
    s3 = pivot - (range_h_l * 1.1 / 2)
    s4 = pivot - (range_h_l * 1.1)
    
    # Align weekly levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    r4_6h = align_htf_to_ltf(prices, df_1w, r4)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20[i]) or np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price returns below R3 or volume drops
            if close[i] < r3_6h[i] or not vol_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns above S3 or volume drops
            if close[i] > s3_6h[i] or not vol_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at R3/S3: price touches extreme level with volume surge
            # Long fade: price <= S3 with volume surge (expect bounce)
            if close[i] <= s3_6h[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short fade: price >= R3 with volume surge (expect rejection)
            elif close[i] >= r3_6h[i] and vol_surge:
                position = -1
                signals[i] = -0.25
            # Breakout continuation: price breaks R4/S4 with volume surge
            # Long breakout: price >= R4 with volume surge
            elif close[i] >= r4_6h[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short breakout: price <= S4 with volume surge
            elif close[i] <= s4_6h[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals
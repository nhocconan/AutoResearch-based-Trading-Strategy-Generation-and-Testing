#!/usr/bin/env python3
"""
4h_12h_camarilla_pivot_volume_v1
Hypothesis: Combines 12h Camarilla pivot levels with 4h price action and volume confirmation to capture institutional breakouts.
In bull markets: captures momentum breaks above resistance with volume.
In bear markets: captures breakdowns below support with volume.
Uses volume filter to avoid false breakouts and reduce whipsaw.
Target: 20-40 trades/year per symbol.
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # S4 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.1/4, etc.
    # We'll use R3/S3 as primary levels
    cam_r3 = close_12h + (high_12h - low_12h) * 1.1 / 4
    cam_s3 = close_12h - (high_12h - low_12h) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_12h, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_12h, cam_s3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(cam_r3_aligned[i]) or 
            np.isnan(cam_s3_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Price breaks Camarilla S3/R3 with volume expansion
        long_entry = (close[i] > cam_r3_aligned[i] and volume_expansion[i])
        short_entry = (close[i] < cam_s3_aligned[i] and volume_expansion[i])
        
        # Exit when price returns to the opposite Camarilla level (mean reversion)
        exit_long = position == 1 and close[i] < cam_s3_aligned[i]
        exit_short = position == -1 and close[i] > cam_r3_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0
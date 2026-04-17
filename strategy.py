#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Volume Confirmation.
Long when price breaks above upper BB(20,2) during low volatility (BBW < 0.05) and 1d volume > 1.5x 20-period average.
Short when price breaks below lower BB(20,2) during low volatility and 1d volume > 1.5x 20-period average.
Exit when price returns to middle BB or volatility expands (BBW > 0.1).
Uses 1d for volume filter, 6h for Bollinger Bands and breakout detection.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-period)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate 6h Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    middle_bb = sma20
    
    # Bollinger Band Width (normalize by middle BB to avoid price level issues)
    bb_width = (upper_bb - lower_bb) / middle_bb
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(avg_volume_1d_aligned[i]) or 
            np.isnan(bb_width[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Squeeze conditions
        is_squeeze = bb_width[i] < 0.05  # low volatility
        is_expansion = bb_width[i] > 0.1  # high volatility (exit condition)
        
        if position == 0:
            # Long: price breaks above upper BB during squeeze with volume confirmation
            if close[i] > upper_bb[i] and is_squeeze and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB during squeeze with volume confirmation
            elif close[i] < lower_bb[i] and is_squeeze and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle BB OR volatility expands
            if close[i] < middle_bb[i] or is_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle BB OR volatility expands
            if close[i] > middle_bb[i] or is_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BBSqueeze_1dVolume_Breakout"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_12h_PriceAction_With_Volume_Momentum
Hypothesis: Trade price action breakouts with volume momentum on 4h using 12h volume confirmation. 
Long when price breaks above recent high with volume surge; short when breaks below recent low with volume surge.
Volume surge confirms institutional participation. Works in bull/bear: breakouts with volume indicate momentum.
Target: 80-150 total trades over 4 years (20-38/year) with position size 0.25.
"""

name = "4h_12h_PriceAction_With_Volume_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h volume average for spike detection
    vol_12h = df_12h['volume'].values
    vol_avg_12h = np.full(len(vol_12h), np.nan)
    for i in range(len(vol_12h)):
        if i >= 19:  # 20-period average
            vol_avg_12h[i] = np.mean(vol_12h[i-19:i+1])
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Need at least 1 day of prior data for high/low calculation
        if i < 6:  # 6 * 4h = 24h worth of 4h bars to have 1 day prior
            continue
            
        # Calculate recent high and low (prior 6 periods = 24 hours)
        recent_high = np.max(prices['high'].iloc[i-6:i])
        recent_low = np.min(prices['low'].iloc[i-6:i])
        
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Volume spike: current volume > 1.5x 12h average volume
        vol_spike = (not np.isnan(vol_avg_12h_aligned[i]) and 
                     current_volume > 1.5 * vol_avg_12h_aligned[i])
        
        if position == 0:
            # Long: price breaks above recent high with volume spike
            if current_close > recent_high and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below recent low with volume spike
            elif current_close < recent_low and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below recent low or volume dries up
            if current_close < recent_low or (not np.isnan(vol_avg_12h_aligned[i]) and 
                                            current_volume < 0.5 * vol_avg_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above recent high or volume dries up
            if current_close > recent_high or (not np.isnan(vol_avg_12h_aligned[i]) and 
                                             current_volume < 0.5 * vol_avg_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1S1_Breakout_Volume_Strength
Hypothesis: Trade breakouts at Camarilla R1/S1 levels on 4h with 12h volume strength confirmation.
Long when price breaks above R1 with volume above 12h median (strong participation); short when breaks below S1 with volume strength.
Uses volume strength vs median (not average) to avoid outlier skew. Works in bull/bear: breaks indicate momentum continuation.
Target: 50-120 total trades over 4 years (13-30/year) with position size 0.25 to limit drawdown.
"""

name = "4h_12h_Camarilla_R1S1_Breakout_Volume_Strength"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for volume strength
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h volume median for strength detection (more robust than mean)
    vol_12h = df_12h['volume'].values
    vol_median_12h = np.full(len(vol_12h), np.nan)
    for i in range(len(vol_12h)):
        if i >= 19:  # 20-period median
            vol_median_12h[i] = np.median(vol_12h[i-19:i+1])
    vol_median_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_median_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 96  # Need 24 hours of prior data for reliable prior day calculation
    
    for i in range(start_idx, n):
        # Need at least 1 day of prior data for Camarilla calculation (24 * 4h bars)
        if i < 96:
            continue
            
        # Calculate Camarilla levels using prior day's OHLC (24 bars back = 1 day prior)
        prior_day_high = np.max(prices['high'].iloc[i-24:i])
        prior_day_low = np.min(prices['low'].iloc[i-24:i])
        prior_day_close = prices['close'].iloc[i-1]  # Previous 4h bar close
        
        # Calculate Camarilla levels
        range_val = prior_day_high - prior_day_low
        if range_val <= 0:
            continue
            
        # Camarilla R1 and S1 levels (core levels for intraday trading)
        r1 = prior_day_close + (range_val * 1.1 / 12)
        s1 = prior_day_close - (range_val * 1.1 / 12)
        
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Volume strength: current volume > 1.2x 12h median volume (avoid noise)
        vol_strong = (not np.isnan(vol_median_12h_aligned[i]) and 
                      current_volume > 1.2 * vol_median_12h_aligned[i])
        
        if position == 0:
            # Long: price breaks above R1 with volume strength
            if current_close > r1 and vol_strong:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume strength
            elif current_close < s1 and vol_strong:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (reversal) or volume weakens significantly
            if current_close < s1 or (not np.isnan(vol_median_12h_aligned[i]) and 
                                    current_volume < 0.6 * vol_median_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reversal) or volume weakens significantly
            if current_close > r1 or (not np.isnan(vol_median_12h_aligned[i]) and 
                                    current_volume < 0.6 * vol_median_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
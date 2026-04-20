#!/usr/bin/env python3
"""
6h_12h_Camarilla_R1S1_Breakout_VolumeFilter
Hypothesis: Trade Camarilla pivot R1/S1 breakouts on 6h with 12h volume confirmation.
Long when price breaks above R1 with volume spike; short when breaks below S1 with volume spike.
Camarilla levels provide intraday support/resistance. Volume filter ensures institutional participation.
Works in bull/bear: breaks indicate momentum continuation, volume confirms validity.
Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25.
"""

name = "6h_12h_Camarilla_R1S1_Breakout_VolumeFilter"
timeframe = "6h"
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
    
    start_idx = 40  # Need enough data for calculations (40*6h=10 days)
    
    for i in range(start_idx, n):
        # Need at least 1 day of prior data for Camarilla calculation (4 bars of 6h)
        if i < 4:
            continue
            
        # Calculate Camarilla levels using prior day's OHLC
        # Look back 4 bars (4*6h = 24h) to get prior day's data
        prior_day_high = np.max(prices['high'].iloc[i-4:i])
        prior_day_low = np.min(prices['low'].iloc[i-4:i])
        prior_day_close = prices['close'].iloc[i-1]  # Previous 6h bar close
        
        # Calculate Camarilla levels
        range_val = prior_day_high - prior_day_low
        if range_val <= 0:
            continue
            
        # Camarilla R1 and S1 levels
        r1 = prior_day_close + (range_val * 1.1 / 12)
        s1 = prior_day_close - (range_val * 1.1 / 12)
        
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Volume spike: current volume > 1.5x 12h average volume
        vol_spike = (not np.isnan(vol_avg_12h_aligned[i]) and 
                     current_volume > 1.5 * vol_avg_12h_aligned[i])
        
        if position == 0:
            # Long: price breaks above R1 with volume spike
            if current_close > r1 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike
            elif current_close < s1 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or volume dries up
            if current_close < s1 or (not np.isnan(vol_avg_12h_aligned[i]) and 
                                    current_volume < 0.5 * vol_avg_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or volume dries up
            if current_close > r1 or (not np.isnan(vol_avg_12h_aligned[i]) and 
                                    current_volume < 0.5 * vol_avg_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 6h_WilliamsAlligator_ElderRay_Trend
# Hypothesis: Williams Alligator (JAW/TEETH/LIPS) defines trend direction, Elder Ray (Bull/Bear Power) filters entries.
# Works in bull/bear: Alligator keeps us in major trends, Elder Ray avoids counter-trend entries during pullbacks.
# Uses 1-day EMA13 for Alligator components and 1-day EMA13/EMA8 for Elder Ray components.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size.

name = "6h_WilliamsAlligator_ElderRay_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Williams Alligator components from 1-day data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Alligator: JAW (13-period SMMA, 8 bars ahead), TEETH (8-period SMMA, 5 bars ahead), LIPS (5-period SMMA, 3 bars ahead)
    # Using EMA as proxy for SMMA with same period for simplicity and better responsiveness
    jaw_1d = np.full_like(close_1d, np.nan)
    teeth_1d = np.full_like(close_1d, np.nan)
    lips_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 13:
        # JAW: 13-period EMA
        jaw_1d[12] = np.mean(close_1d[0:13])
        for i in range(13, len(close_1d)):
            jaw_1d[i] = (jaw_1d[i-1] * 12 + close_1d[i]) / 13
        
        # TEETH: 8-period EMA
        teeth_1d[7] = np.mean(close_1d[0:8])
        for i in range(8, len(close_1d)):
            teeth_1d[i] = (teeth_1d[i-1] * 7 + close_1d[i]) / 8
        
        # LIPS: 5-period EMA
        lips_1d[4] = np.mean(close_1d[0:5])
        for i in range(5, len(close_1d)):
            lips_1d[i] = (lips_1d[i-1] * 4 + close_1d[i]) / 5
    
    # Align Alligator components to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Using 1-day EMA13 as the reference
    ema13_1d = jaw_1d  # JAW is already 13-period EMA
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Ensure EMA13 is ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bullish alignment (Lips > Teeth > Jaw) AND Bull Power > 0
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i] and 
                bull_power_1d_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish alignment (Lips < Teeth < Jaw) AND Bear Power < 0
            elif (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i] and 
                  bear_power_1d_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish alignment OR Bull Power turns negative
            if (lips_1d_aligned[i] < teeth_1d_aligned[i] or 
                bull_power_1d_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish alignment OR Bear Power turns positive
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] or 
                bear_power_1d_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
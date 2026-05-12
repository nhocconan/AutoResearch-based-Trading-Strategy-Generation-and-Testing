#!/usr/bin/env python3
name = "6h_ElderRay_Alligator_Trend_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Elder Ray and Alligator
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
    # Using EMA as approximation for SMMA
    jaw_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Shift to align with Alligator rules
    jaw_1d_shifted = np.roll(jaw_1d, 8)
    teeth_1d_shifted = np.roll(teeth_1d, 5)
    lips_1d_shifted = np.roll(lips_1d, 3)
    # Fill initial values
    jaw_1d_shifted[:8] = jaw_1d[0]
    teeth_1d_shifted[:5] = teeth_1d[0]
    lips_1d_shifted[:3] = lips_1d[0]
    
    # Align all indicators to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d_shifted)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d_shifted)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d_shifted)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, Lips > Jaw > Teeth (bullish alignment), volume filter
            if (bull_power_1d_aligned[i] > 0 and bear_power_1d_aligned[i] < 0 and 
                lips_1d_aligned[i] > jaw_1d_aligned[i] > teeth_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, Lips < Jaw < Teeth (bearish alignment), volume filter
            elif (bear_power_1d_aligned[i] < 0 and bull_power_1d_aligned[i] > 0 and 
                  lips_1d_aligned[i] < jaw_1d_aligned[i] < teeth_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or alignment breaks
            if (bull_power_1d_aligned[i] <= 0 or not (lips_1d_aligned[i] > jaw_1d_aligned[i] > teeth_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or alignment breaks
            if (bear_power_1d_aligned[i] >= 0 or not (lips_1d_aligned[i] < jaw_1d_aligned[i] < teeth_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
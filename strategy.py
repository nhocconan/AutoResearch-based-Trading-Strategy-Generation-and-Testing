#!/usr/bin/env python3
name = "6h_Alligator_ElderRay_Trend_v1"
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
    
    # Get 12h data for Alligator and Elder Ray
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Williams Alligator (13,8,5 SMAs)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    jaw = pd.Series(high_12h).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(high_12h).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(high_12h).rolling(window=5, min_periods=5).mean().values   # Green line
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 12h close)
    close_12h = df_12h['close'].values
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_12h - ema13_12h
    bear_power = low_12h - ema13_12h
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20)  # Need 13 for Alligator/Elder Ray, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips above Teeth above Jaw (bullish alignment) AND Bull Power > 0 AND volume
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                bull_power_aligned[i] > 0 and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips below Teeth below Jaw (bearish alignment) AND Bear Power < 0 AND volume
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  bear_power_aligned[i] < 0 and volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long: Alligator alignment breaks down OR Bull Power turns negative
                if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or bull_power_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Alligator alignment breaks down OR Bear Power turns positive
                if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or bear_power_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
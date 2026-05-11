#!/usr/bin/env python3
name = "6h_ElderRay_Alligator_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate 1d EMA13 and EMA8 for Alligator (Williams Alligator)
    df_1d = get_htf_data(prices, '1d')
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema8_1d = pd.Series(df_1d['close']).ewm(span=8, min_periods=8, adjust=False).mean().values
    
    # Align 1d EMA values to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema8_1d_aligned = align_htf_to_ltf(prices, df_1d, ema8_1d)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Smooth Bull/Bear Power with 5-period EMA
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, min_periods=5, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    # Volume filter: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(ema8_1d_aligned[i]) or 
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        # Bullish: Bull Power > 0, Bear Power < 0, EMA8 > EMA13 (Alligator aligned up), Volume OK
        bullish = (bull_power_smooth[i] > 0 and 
                   bear_power_smooth[i] < 0 and 
                   ema8_1d_aligned[i] > ema13_1d_aligned[i] and 
                   volume_ok[i])
        
        # Bearish: Bear Power < 0, Bull Power < 0, EMA8 < EMA13 (Alligator aligned down), Volume OK
        bearish = (bear_power_smooth[i] < 0 and 
                   bull_power_smooth[i] < 0 and 
                   ema8_1d_aligned[i] < ema13_1d_aligned[i] and 
                   volume_ok[i])
        
        if position == 0:
            # Long entry
            if bullish:
                signals[i] = 0.25
                position = 1
            # Short entry
            elif bearish:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Bull Power turns negative OR Alligator misaligns (EMA8 < EMA13)
                if bull_power_smooth[i] <= 0 or ema8_1d_aligned[i] <= ema13_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: Bear Power turns positive OR Alligator misaligns (EMA8 > EMA13)
                if bear_power_smooth[i] >= 0 or ema8_1d_aligned[i] >= ema13_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
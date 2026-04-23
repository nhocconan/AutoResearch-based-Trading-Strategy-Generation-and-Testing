#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike with 1d Trend Filter
- Uses Williams Alligator (Jaw/Teeth/Lips) from 4h for trend direction and momentum
- Elder Ray (Bull/Bear Power) from 1d confirms trend strength
- Volume confirmation (> 2.0x 20-period average) filters weak signals
- Only trade in direction of 1d EMA50 trend: long when price > EMA50, short when price < EMA50
- Designed for 4h timeframe targeting 15-25 trades/year (60-100 over 4 years)
- Works in both bull and bear markets by following the 1d EMA50 trend filter
"""

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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Williams Alligator on 4h: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean()
    jaw = jaw.shift(8)  # Smoothed with 8-period delay
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean()
    teeth = teeth.shift(5)  # Smoothed with 5-period delay
    lips = pd.Series(close).ewm(span=5, adjust=False).mean()
    lips = lips.shift(3)  # Smoothed with 3-period delay
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # need 1d EMA50, 1d Elder Ray, Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(jaw_values[i]) or 
            np.isnan(teeth_values[i]) or 
            np.isnan(lips_values[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator aligned (Lips > Teeth > Jaw) AND Bull Power > 0 AND price > 1d EMA50 AND volume spike
            if (lips_values[i] > teeth_values[i] > jaw_values[i] and 
                bull_power_1d_aligned[i] > 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned (Lips < Teeth < Jaw) AND Bear Power < 0 AND price < 1d EMA50 AND volume spike
            elif (lips_values[i] < teeth_values[i] < jaw_values[i] and 
                  bear_power_1d_aligned[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator reverses OR price crosses 1d EMA50
            exit_signal = False
            
            if position == 1:
                # Exit long when Alligator reverses (Lips < Teeth) OR price < 1d EMA50
                if lips_values[i] < teeth_values[i] or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when Alligator reverses (Lips > Teeth) OR price > 1d EMA50
                if lips_values[i] > teeth_values[i] or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_VolumeSpike_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0
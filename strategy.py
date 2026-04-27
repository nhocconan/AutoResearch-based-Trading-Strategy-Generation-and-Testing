#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Elder Ray components (Bull/Bear Power) from 1d data
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d EMA13 for trend filter (same as used in Elder Ray)
    ema13_6h = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema13_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Bull Power > 0 (strong buying pressure) with price above EMA13 and volume spike
        # 2. Bear Power turning positive from negative (momentum shift) with volume spike
        long_condition = (bull_power_6h[i] > 0 and close[i] > ema13_6h[i] and volume_spike[i]) or \
                         (bear_power_6h[i] > 0 and bear_power_6h[i-1] <= 0 and volume_spike[i])
        
        # Short conditions:
        # 1. Bear Power < 0 (strong selling pressure) with price below EMA13 and volume spike
        # 2. Bull Power turning negative from positive (momentum shift) with volume spike
        short_condition = (bear_power_6h[i] < 0 and close[i] < ema13_6h[i] and volume_spike[i]) or \
                          (bull_power_6h[i] < 0 and bull_power_6h[i-1] >= 0 and volume_spike[i])
        
        if long_condition:
            signals[i] = 0.25
            position = 1
        elif short_condition:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite power signal or price crosses EMA13
        elif position == 1 and (bear_power_6h[i] > 0 or close[i] < ema13_6h[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bull_power_6h[i] < 0 or close[i] > ema13_6h[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_BullBearPower_EMA13_Volume1.5x_1d"
timeframe = "6h"
leverage = 1.0
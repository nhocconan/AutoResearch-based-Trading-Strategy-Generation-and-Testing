#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_Swing_Double_EMA"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 and EMA48 for 6h trend
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_48 = pd.Series(close).ewm(span=48, adjust=False, min_periods=48).mean().values
    
    # Get 1d data for Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d close for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Swing detection: swing high when bear power turns negative after positive
    # swing low when bull power turns negative after positive
    swing_high = (bull_power_aligned > 0) & (np.roll(bull_power_aligned, 1) <= 0)
    swing_low = (bear_power_aligned < 0) & (np.roll(bear_power_aligned, 1) >= 0)
    
    # Handle first element
    swing_high[0] = False
    swing_low[0] = False
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_13[i]) or np.isnan(ema_48[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bull power turning positive (swing low) + EMA13 > EMA48 + volume filter
            long_cond = swing_low[i] and (ema_13[i] > ema_48[i]) and volume_filter[i]
            # Short: bear power turning negative (swing high) + EMA13 < EMA48 + volume filter
            short_cond = swing_high[i] and (ema_13[i] < ema_48[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bear power turning negative or EMA13 < EMA48
            if bear_power_aligned[i] < 0 or ema_13[i] < ema_48[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bull power turning positive or EMA13 > EMA48
            if bull_power_aligned[i] > 0 or ema_13[i] > ema_48[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
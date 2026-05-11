#!/usr/bin/env python3
name = "6h_ElderRay_BullBear_1dTrend_WeakFilter"
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
    
    # 1d data for EMA13 trend filter and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1w data for trend filter (optional confirmation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA13 on 1d close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate EMA13 on 1w close
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power on 1d
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align 1d indicators to 6h
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Align 1w EMA13 to 6h
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Volume spike (24-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure EMA13 is ready
    
    for i in range(start_idx, n):
        if np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema13_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price above 1d EMA13, price above 1w EMA13, volume spike
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                close[i] > ema13_1d_aligned[i] and 
                close[i] > ema13_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, price below 1d EMA13, price below 1w EMA13, volume spike
            elif (bull_power_aligned[i] < 0 and 
                  bear_power_aligned[i] > 0 and 
                  close[i] < ema13_1d_aligned[i] and 
                  close[i] < ema13_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or price below 1d EMA13
            if bull_power_aligned[i] <= 0 or close[i] < ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or price above 1d EMA13
            if bear_power_aligned[i] >= 0 or close[i] > ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
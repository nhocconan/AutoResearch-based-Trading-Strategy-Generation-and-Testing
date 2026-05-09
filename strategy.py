#!/usr/bin/env python3
"""
6h_ElderRay_1dTrend_Volume_Filter
Hypothesis: Combine Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume spikes.
- Bull Power = High - EMA13 (1d); Bear Power = EMA13 (1d) - Low
- Go long when Bull Power > 0, price > 1d EMA13, and volume > 1.5x 20-period average
- Go short when Bear Power > 0, price < 1d EMA13, and volume > 1.5x 20-period average
- Exit when power turns negative or price crosses EMA13
- Works in both bull and bear markets via trend filter and volatility-based entry
"""

name = "6h_ElderRay_1dTrend_Volume_Filter"
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
    
    # 6-period EMA for entry timing (fast)
    ema_fast = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Get 1d data for EMA13 and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for trend
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components on 1d
    # Bull Power = High - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    # Bear Power = EMA13 - Low
    bear_power = ema13_1d - df_1d['low'].values
    
    # Align 1d indicators to 6h timeframe (waits for daily close)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_fast[i]) or np.isnan(ema13_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power positive, price above EMA13, volume spike
            if (bull_power_aligned[i] > 0 and 
                close[i] > ema13_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power positive, price below EMA13, volume spike
            elif (bear_power_aligned[i] > 0 and 
                  close[i] < ema13_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative OR price crosses below EMA13
            if (bull_power_aligned[i] <= 0) or (close[i] < ema13_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns negative OR price crosses above EMA13
            if (bear_power_aligned[i] <= 0) or (close[i] > ema13_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
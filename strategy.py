#!/usr/bin/env python3
"""
6h_ElderRay_BullPower_Trend_With_Volume_Filter
Hypothesis: Elder Ray Bull/Bear power from daily timeframe indicates institutional buying/selling pressure.
Go long when daily Bull Power > 0 and price above 13-period EMA, short when Bear Power < 0 and price below EMA.
Volume filter ensures breakout strength. Designed for 6h timeframe to capture multi-day trends with low frequency.
Works in both bull (sustained Bull Power > 0) and bear (sustained Bear Power < 0) markets.
"""

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
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA of daily close
    ema13_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 13:
        ema13_1d[12] = np.mean(close_1d[0:13])
        alpha = 2 / (13 + 1)
        for i in range(13, len(close_1d)):
            ema13_1d[i] = close_1d[i] * alpha + ema13_1d[i-1] * (1 - alpha)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 20-period volume MA on 6h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily Bull Power > 0 (buying pressure) + price above EMA + volume spike
            if (bull_power_1d_aligned[i] > 0 and 
                close[i] > ema13_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: daily Bear Power < 0 (selling pressure) + price below EMA + volume spike
            elif (bear_power_1d_aligned[i] < 0 and 
                  close[i] < ema13_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative OR price breaks below EMA
            if (bull_power_1d_aligned[i] <= 0 or close[i] < ema13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive OR price breaks above EMA
            if (bear_power_1d_aligned[i] >= 0 or close[i] > ema13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullPower_Trend_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0
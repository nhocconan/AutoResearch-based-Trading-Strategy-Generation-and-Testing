#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend
Hypothesis: Elder Ray bull/bear power with 1-day trend filter and volume confirmation.
Works in both bull and bear markets by capturing trend continuation from daily extremes.
Targets 20-40 trades/year on 6h timeframe with disciplined risk control.
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
    
    # Calculate 1-day EMA13 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1-day EMA13
    ema13_1d = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        if i == 13:
            ema13_1d[i] = np.mean(close_1d[0:14])
        else:
            k = 2 / (13 + 1)
            ema13_1d[i] = close_1d[i] * k + ema13_1d[i-1] * (1 - k)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 13-period EMA for Elder Ray calculation
    ema13 = np.full(n, np.nan)
    for i in range(13, n):
        if i == 13:
            ema13[i] = np.mean(close[0:14])
        else:
            k = 2 / (13 + 1)
            ema13[i] = close[i] * k + ema13[i-1] * (1 - k)
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA
    bear_power = low - ema13   # Bear Power = Low - EMA
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, price above 1-day EMA, volume confirmation
            if (bull_power[i] > 0 and close[i] > ema13_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, price below 1-day EMA, volume confirmation
            elif (bear_power[i] < 0 and close[i] < ema13_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative or price breaks below 1-day EMA
            if (bull_power[i] <= 0 or close[i] < ema13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive or price breaks above 1-day EMA
            if (bear_power[i] >= 0 or close[i] > ema13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
leverage = 1.0
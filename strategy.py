#!/usr/bin/env python3
"""
6h Elder Ray Power with 1d Trend Filter and Volume Spike.
Long when bull power > 0, bear power crosses above zero, 1d trend up, and volume spike.
Short when bear power < 0, bull power crosses below zero, 1d trend down, and volume spike.
Exit when power crosses zero or 1d trend reverses.
Designed for low frequency (12-30 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.empty_like(close_1d, dtype=np.float64)
    ema_1d.fill(np.nan)
    for i in range(12, len(close_1d)):
        ema_1d[i] = np.mean(close_1d[i-12:i+1])  # Simple MA approximation
    
    # Align 1d EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 13-period EMA for Elder Ray (using 6h data)
    ema13 = np.empty_like(close, dtype=np.float64)
    ema13.fill(np.nan)
    for i in range(12, n):
        ema13[i] = np.mean(close[i-12:i+1])
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Volume filter: volume > 1.5x average
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA13 (13), EMA1d (13), volume MA (20)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        bull = bull_power[i]
        bear = bear_power[i]
        trend_1d = ema_1d_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Bull: bull power > 0, bear power crosses above zero, 1d trend up, volume spike
            if bull > 0 and bear > 0 and bear_power[i-1] <= 0 and trend_1d > close[i] and vol_filter:
                signals[i] = size
                position = 1
            # Bear: bear power < 0, bull power crosses below zero, 1d trend down, volume spike
            elif bear < 0 and bull < 0 and bull_power[i-1] >= 0 and trend_1d < close[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bull power <= 0 or 1d trend turns down
            if bull <= 0 or trend_1d < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bear power >= 0 or 1d trend turns up
            if bear >= 0 or trend_1d > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRayPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0
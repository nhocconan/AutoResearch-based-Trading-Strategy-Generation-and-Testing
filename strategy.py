#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d Trend + Volume Spike.
Long when Bull Power > 0, Bear Power < 0, 1d EMA50 up, volume spike.
Short when Bull Power < 0, Bear Power > 0, 1d EMA50 down, volume spike.
Exit when either power crosses zero or 1d trend reverses.
Elder Ray measures bull/bear strength relative to EMA; works in trends and reversals.
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
    
    # Get 1d data for EMA trend and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.empty_like(close_1d, dtype=np.float64)
    ema_1d.fill(np.nan)
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_1d[i] = close_1d[i]
        elif np.isnan(ema_1d[i-1]):
            ema_1d[i] = close_1d[i]
        else:
            ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Elder Ray components on 1d
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13_1d = np.empty_like(close_1d, dtype=np.float64)
    ema13_1d.fill(np.nan)
    alpha13 = 2.0 / (13 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema13_1d[i] = close_1d[i]
        elif np.isnan(ema13_1d[i-1]):
            ema13_1d[i] = close_1d[i]
        else:
            ema13_1d[i] = alpha13 * close_1d[i] + (1 - alpha13) * ema13_1d[i-1]
    
    bull_power = high - ema13_1d
    bear_power = low - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume filter: volume > 1.8x average (to avoid false signals)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily EMA(50), EMA(13), volume MA (20)
    start_idx = max(19, 13, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        trend_1d = ema_1d_aligned[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Bull: Bull Power > 0, Bear Power < 0, 1d trend up, volume spike
            if bull > 0 and bear < 0 and price_now > trend_1d and vol_filter:
                signals[i] = size
                position = 1
            # Bear: Bull Power < 0, Bear Power > 0, 1d trend down, volume spike
            elif bull < 0 and bear > 0 and price_now < trend_1d and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power crosses below zero OR Bear Power crosses above zero OR trend down
            if bull < 0 or bear > 0 or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bull Power crosses above zero OR Bear Power crosses below zero OR trend up
            if bull > 0 or bear < 0 or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRayPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0
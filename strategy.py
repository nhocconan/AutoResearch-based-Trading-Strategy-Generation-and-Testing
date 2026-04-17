#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_v1
4h Donchian breakout with 12h EMA trend filter and volume confirmation.
Long on breakout above upper band with uptrend, short on breakdown below lower band with downtrend.
Volume filter requires >1.5x average volume to confirm breakout strength.
Designed for low-frequency, high-conviction trades to minimize fee drag.
Target: 20-50 trades per year per symbol.
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
    
    # === 4h Donchian Channel (20-period) ===
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            highest_high[i] = np.max(high[i-lookback+1:i+1])
            lowest_low[i] = np.min(low[i-lookback+1:i+1])
        elif i > 0:
            highest_high[i] = np.max(high[0:i+1])
            lowest_low[i] = np.min(low[0:i+1])
        else:
            highest_high[i] = high[i]
            lowest_low[i] = low[i]
    
    # === 12h EMA (34-period) for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA with proper smoothing
    ema_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        # Initialize with SMA
        ema_12h[33] = np.mean(close_12h[0:34])
        # Calculate EMA for remaining periods
        alpha = 2.0 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(n):
        if i >= 19:  # 20 periods including current
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[0:i+1])
        else:
            vol_ma_20[i] = volume[i]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = max(50, lookback)
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: breakout above upper Donchian band + uptrend (price > EMA) + volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > ema_12h_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: breakdown below lower Donchian band + downtrend (price < EMA) + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_12h_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below lower Donchian band OR trend reverses
            if (close[i] < lowest_low[i] or 
                close[i] < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper Donchian band OR trend reverses
            if (close[i] > highest_high[i] or 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0
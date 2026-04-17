#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_Trend_v1
Breakout of 20-period Donchian channel on 12h timeframe with volume confirmation
and daily trend filter. Designed to capture trends in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Donchian Channel (20-period) ===
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:  # 20-period window
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
    
    # === 12h Volume Confirmation (20-period average) ===
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:  # 20-period window
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1]) if i > 0 else volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === 1d EMA34 (Trend Filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA34 calculation
    ema_34 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34[33] = np.mean(close_1d[:34])  # Simple average for first value
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34[i] = close_1d[i] * alpha + ema_34[i-1] * (1 - alpha)
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_confirm[i]) or 
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian band AND volume confirmation AND price above daily EMA34
            if (close[i] > highest_high[i] and 
                vol_confirm[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian band AND volume confirmation AND price below daily EMA34
            elif (close[i] < lowest_low[i] and 
                  vol_confirm[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below lower Donchian band (re-entry level)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper Donchian band (re-entry level)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0
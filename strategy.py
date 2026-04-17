#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Trend_v1
Donchian(20) breakout + volume confirmation + EMA34 trend filter on 12h.
Breakouts in direction of higher timeframe trend (1d EMA34).
Volume filter requires 1.5x average volume to avoid false breakouts.
Designed for low trade frequency (~20-40 trades/year) to minimize fee drag.
Works in both bull and bear markets by following the trend.
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
    
    # === 1d EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34 = np.zeros_like(close_1d)
    ema_34[:] = np.nan
    alpha = 2 / (34 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_34[i] = close_1d[i]
        elif not np.isnan(close_1d[i]):
            if np.isnan(ema_34[i-1]):
                ema_34[i] = close_1d[i]
            else:
                ema_34[i] = close_1d[i] * alpha + ema_34[i-1] * (1 - alpha)
    
    # === 12h Donchian channels (20-period) ===
    donch_high = np.full_like(high, np.nan)
    donch_low = np.full_like(low, np.nan)
    for i in range(n):
        if i >= 19:
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
        elif i > 0:
            donch_high[i] = np.max(high[0:i+1])
            donch_low[i] = np.min(low[0:i+1])
        else:
            donch_high[i] = high[i]
            donch_low[i] = low[i]
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[0:i+1])
        else:
            vol_ma_20[i] = volume[i]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high AND above 1d EMA34 (uptrend) AND volume confirmation
            if (close[i] > donch_high[i] and 
                close[i] > ema_34_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low AND below 1d EMA34 (downtrend) AND volume confirmation
            elif (close[i] < donch_low[i] and 
                  close[i] < ema_34_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below Donchian low
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian high
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0
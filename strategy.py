#!/usr/bin/env python3
"""
12h_PriceChannel_VolumeRegime_v1
Breakout strategy on 12h timeframe using Donchian channels (20) for entry,
filtered by 1-day trend direction (EMA50) and volume confirmation.
Designed for low trade frequency (<30/year) to minimize fee drag.
Works in both bull and bear markets by trading breakouts in the direction of higher timeframe trend.
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
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily close
    ema_50 = np.zeros_like(close_1d)
    ema_50[0] = close_1d[0]
    alpha = 2 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    
    # === 12h Donchian Channel (20) ===
    donch_high = np.full_like(high, np.nan)
    donch_low = np.full_like(low, np.nan)
    
    for i in range(n):
        if i >= 20:
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
        else:
            donch_high[i] = np.max(high[:i+1])
            donch_low[i] = np.min(low[:i+1])
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[:i+1]) if i > 0 else volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === Align 1d EMA50 to 12h timeframe ===
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high AND above 1d EMA50 (uptrend) AND volume confirmation
            if (close[i] > donch_high[i] and 
                close[i] > ema_50_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low AND below 1d EMA50 (downtrend) AND volume confirmation
            elif (close[i] < donch_low[i] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below Donchian low OR below 1d EMA50
            if (close[i] < donch_low[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR above 1d EMA50
            if (close[i] > donch_high[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PriceChannel_VolumeRegime_v1"
timeframe = "12h"
leverage = 1.0
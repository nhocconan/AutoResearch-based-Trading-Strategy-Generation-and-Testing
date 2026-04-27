#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_12hTrend
Breakout strategy using Donchian channels (20) on 4h with volume confirmation and 12h trend filter.
Long when price breaks above upper Donchian + volume spike + 12h EMA uptrend.
Short when price breaks below lower Donchian + volume spike + 12h EMA downtrend.
Exit when price crosses back through middle Donchian or trend fails.
Uses 25% position size. Target: 20-50 trades/year per symbol.
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
    
    # Donchian Channel (20) on 4h
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower[i] = np.min(low[i - donchian_period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Volume spike detection (volume > 2x 20-period average)
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_period + 1:i + 1])
    volume_spike = np.zeros(n, dtype=bool)
    for i in range(vol_period - 1, n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            volume_spike[i] = volume[i] > (2.0 * vol_ma[i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                          ema_12h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align 12h EMA50 to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, volume MA, and EMA
    start_idx = max(donchian_period - 1, vol_period - 1, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema12h_val = ema_12h_aligned[i]
        
        if position == 0:
            # Long: break above upper Donchian + volume spike + 12h EMA uptrend
            if (price > upper[i] and vol_spike and price > ema12h_val):
                signals[i] = size
                position = 1
            # Short: break below lower Donchian + volume spike + 12h EMA downtrend
            elif (price < lower[i] and vol_spike and price < ema12h_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below middle Donchian or trend fails
            if price < middle[i] or price < ema12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above middle Donchian or trend fails
            if price > middle[i] or price > ema12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_12hTrend"
timeframe = "4h"
leverage = 1.0
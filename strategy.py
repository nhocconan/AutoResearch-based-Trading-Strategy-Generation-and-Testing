#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_TrendFilter_v2
Donchian(20) breakout + volume spike + EMA trend filter on 4h timeframe.
Uses 1d EMA34 as trend filter to align with higher timeframe direction,
volume spike (2x 20-period average) for confirmation, and Donchian breakouts for entry.
Designed to capture strong momentum moves while avoiding choppy markets.
Target: 80-180 total trades over 4 years (20-45/year).
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
    
    # === Donchian Channel (20-period) ===
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(len(high)):
        if i >= 19:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
        else:
            highest_high[i] = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low[i] = np.min(low[:i+1]) if i > 0 else low[i]
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    
    vol_confirm = volume > vol_ma_20 * 2.0  # volume spike: 2x average
    
    # === 1d EMA34 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 33:
            if i == 33:
                ema_34[i] = np.mean(close_1d[:34])
            else:
                ema_34[i] = (close_1d[i] * 2 + ema_34[i-1] * 33) / 34
        else:
            ema_34[i] = np.nan
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 40
    
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
            # Long: price breaks above Donchian high AND volume confirmation AND uptrend (price > EMA34)
            if (close[i] > highest_high[i] and 
                vol_confirm[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low AND volume confirmation AND downtrend (price < EMA34)
            elif (close[i] < lowest_low[i] and 
                  vol_confirm[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below Donchian low OR volume drops
            if (close[i] < lowest_low[i] or 
                not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR volume drops
            if (close[i] > highest_high[i] or 
                not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_TrendFilter_v2"
timeframe = "4h"
leverage = 1.0
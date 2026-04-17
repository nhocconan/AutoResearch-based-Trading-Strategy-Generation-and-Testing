#!/usr/bin/env python3
"""
4h_DonchianBreakout_Volume_Trend_v1
Donchian(20) breakout + volume confirmation + 12h EMA34 trend filter.
Long when price breaks above upper Donchian band with volume spike and 12h EMA34 up.
Short when price breaks below lower Donchian band with volume spike and 12h EMA34 down.
Exit on opposite Donchian band touch or trend reversal.
Designed for 4h timeframe to capture medium-term trends with proper filtering to avoid overtrading.
Target: 50-150 total trades over 4 years (12-37/year).
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
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(n):
        if i >= 19:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
        else:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
    
    # === 4h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(n):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.nan
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === 12h EMA34 trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = np.full_like(close_12h, np.nan)
    for i in range(len(close_12h)):
        if i >= 34:
            if i == 34:
                ema_12h[i] = np.mean(close_12h[1:35])
            else:
                ema_12h[i] = ema_12h[i-1] * (33/35) + close_12h[i] * (2/35)
        else:
            ema_12h[i] = np.nan
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_confirm[i]) or 
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian band + volume spike + 12h EMA34 up
            if (close[i] > highest_high[i] and 
                vol_confirm[i] and 
                ema_12h_aligned[i] > ema_12h_aligned[i-1]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian band + volume spike + 12h EMA34 down
            elif (close[i] < lowest_low[i] and 
                  vol_confirm[i] and 
                  ema_12h_aligned[i] < ema_12h_aligned[i-1]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price touches lower Donchian band OR 12h EMA34 turns down
            if (close[i] < lowest_low[i] or 
                ema_12h_aligned[i] < ema_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches upper Donchian band OR 12h EMA34 turns up
            if (close[i] > highest_high[i] or 
                ema_12h_aligned[i] > ema_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0
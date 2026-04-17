#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Trend_v1
4h Donchian(20) breakout with volume confirmation and daily trend filter.
Breakout above 20-period high with volume surge and daily close > daily EMA50.
Exit when price closes below 20-period low or volume drops below average.
Designed to capture strong trends with controlled trade frequency.
Target: 75-200 total trades over 4 years (19-50/year).
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
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    for i in range(n):
        if i >= 20:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
        else:
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
    
    # === 4h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(n):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 50:
            if i == 50:
                ema_50[i] = np.mean(close_1d[:51])
            else:
                ema_50[i] = ema_50[i-1] * (49/51) + 2 * close_1d[i] / 51
        else:
            ema_50[i] = np.nan
    
    # === Align 1d EMA50 to 4h timeframe ===
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: breakout above Donchian high + volume surge + daily uptrend
            if (close[i] > donchian_high[i] and 
                volume[i] > vol_ma_20[i] * 1.5 and 
                ema_50_aligned[i] > 0):  # daily close > EMA50
                signals[i] = 0.25
                position = 1
                continue
            # Short: breakdown below Donchian low + volume surge + daily downtrend
            elif (close[i] < donchian_low[i] and 
                  volume[i] > vol_ma_20[i] * 1.5 and 
                  ema_50_aligned[i] < 0):  # daily close < EMA50
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: close below Donchian low OR volume drops below average
            if (close[i] < donchian_low[i] or 
                volume[i] < vol_ma_20[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above Donchian high OR volume drops below average
            if (close[i] > donchian_high[i] or 
                volume[i] < vol_ma_20[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0
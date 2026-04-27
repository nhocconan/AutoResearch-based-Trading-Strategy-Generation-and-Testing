#!/usr/bin/env python3
"""
4h Camarilla Pivot R4/S4 Breakout with 1d EMA34 Trend and Volume Spike.
Long when price breaks above R4 + 1d trend up + volume spike.
Short when price breaks below S4 + 1d trend down + volume spike.
Exit when price returns to central pivot (PP) or trend reverses.
Uses tighter R4/S4 levels (1.5 multiplier) to reduce false breakouts and trades.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels with tighter multiplier (1.5 instead of 1.1)
    r4 = pp + (range_1d * 1.5)   # R4 = PP + 1.5 * (H-L)
    s4 = pp - (range_1d * 1.5)   # S4 = PP - 1.5 * (H-L)
    
    # Align daily Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Get 1d EMA(34) for trend filter
    ema_1d = np.empty_like(close_1d, dtype=np.float64)
    ema_1d.fill(np.nan)
    alpha = 2.0 / (34 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_1d[i] = close_1d[i]
        elif np.isnan(ema_1d[i-1]):
            ema_1d[i] = close_1d[i]
        else:
            ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: volume > 2.0x average (to avoid false breakouts)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily pivot + volume MA (20) + 1d EMA (34)
    start_idx = max(1, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        r4_level = r4_aligned[i]
        s4_level = s4_aligned[i]
        pp_level = pp_aligned[i]
        trend_1d = ema_1d_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above R4 + 1d trend up + volume spike
            if price_now > r4_level and price_now > trend_1d and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below S4 + 1d trend down + volume spike
            elif price_now < s4_level and price_now < trend_1d and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to central pivot (PP) or 1d trend turns down
            if price_now < pp_level or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to central pivot (PP) or 1d trend turns up
            if price_now > pp_level or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R4S4_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
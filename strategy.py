#!/usr/bin/env python3
"""
6h Donchian Breakout with 12h Trend Filter and Volume Spike.
Long when price breaks above Donchian high (20) + 12h trend up + volume spike.
Short when price breaks below Donchian low (20) + 12h trend down + volume spike.
Exit when price returns to Donchian midpoint or trend reverses.
Designed to generate 12-37 trades/year per symbol (50-150 total over 4 years).
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = np.empty_like(close_12h, dtype=np.float64)
    ema_12h.fill(np.nan)
    alpha = 2.0 / (34 + 1)
    for i in range(len(close_12h)):
        if i == 0:
            ema_12h[i] = close_12h[i]
        elif np.isnan(ema_12h[i-1]):
            ema_12h[i] = close_12h[i]
        else:
            ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align 12h EMA to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Volume filter: volume > 1.8x average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + volume MA (20) + 12h EMA (34)
    start_idx = max(19, 19, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        dc_mid = donchian_mid[i]
        trend_12h = ema_12h_aligned[i]
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above Donchian high + 12h trend up + volume spike
            if price_now > dc_high and price_now > trend_12h and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below Donchian low + 12h trend down + volume spike
            elif price_now < dc_low and price_now < trend_12h and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or 12h trend turns down
            if price_now < dc_mid or price_now < trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or 12h trend turns up
            if price_now > dc_mid or price_now > trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_Breakout_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0
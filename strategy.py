#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and Trend Filter.
Long when: 1) Price breaks above Donchian upper (20-period high), 2) Volume > 1.5x 20-period average, 3) Price > 50-period SMA (bullish trend).
Short when: 1) Price breaks below Donchian lower (20-period low), 2) Volume > 1.5x 20-period average, 3) Price < 50-period SMA (bearish trend).
Exit when price returns to Donchian middle (mean reversion) or trend reverses.
Designed for 4h timeframe: targets 75-200 total trades over 4 years (19-50/year).
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
    
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # 50-period SMA for trend filter
    sma_50 = np.full(n, np.nan)
    for i in range(49, n):
        sma_50[i] = np.mean(close[i-49:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), volume MA (20), SMA (50)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(sma_50[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        trend = sma_50[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper + bullish trend + volume spike
            if price > upper and price > trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower + bearish trend + volume spike
            elif price < lower and price < trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to mid (mean reversion) or trend turns bearish
            if price <= mid or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to mid (mean reversion) or trend turns bullish
            if price >= mid or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h Donchian Breakout + Daily Trend Filter + Volume Spike (Novel)
Long when price breaks above 20-period Donchian upper band + daily EMA50 up + volume spike.
Short when price breaks below 20-period Donchian lower band + daily EMA50 down + volume spike.
Exit when price crosses back below/above Donchian middle band.
Designed for low frequency (15-30 trades/year) to minimize fee drag.
Uses Donchian channels as price channel structure with daily trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.empty_like(close_1d, dtype=np.float64)
    ema50_1d.fill(np.nan)
    for i in range(49, len(close_1d)):
        if i == 49:
            ema50_1d[i] = np.mean(close_1d[:50])
        else:
            ema50_1d[i] = (close_1d[i] * 2/51) + (ema50_1d[i-1] * 49/51)
    
    # Align EMA50 to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-period Donchian channels on 6h data
    upper = np.empty_like(high, dtype=np.float64)
    lower = np.empty_like(low, dtype=np.float64)
    middle = np.empty_like(close, dtype=np.float64)
    upper.fill(np.nan)
    lower.fill(np.nan)
    middle.fill(np.nan)
    
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    # Volume filter: volume > 2.0x average (to avoid false breakouts)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + EMA50 (50) + volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        upper_band = upper[i]
        lower_band = lower[i]
        middle_band = middle[i]
        trend = ema50_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above upper band + daily trend up + volume spike
            if price_now > upper_band and price_now > trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below lower band + daily trend down + volume spike
            elif price_now < lower_band and price_now < trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below middle band
            if price_now < middle_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above middle band
            if price_now > middle_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h Trend Filter and Volume Spike.
Long when price breaks above Donchian(20) upper band + 12h EMA50 up + volume spike.
Short when price breaks below Donchian(20) lower band + 12h EMA50 down + volume spike.
Exit when price crosses back inside Donchian channel or trend reverses.
Designed for low frequency (20-40 trades/year) to minimize fee drag.
Uses Donchian channel for breakout signals with 12h EMA trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, window):
    """Calculate Donchian upper and lower bands"""
    upper = np.full_like(high, np.nan, dtype=np.float64)
    lower = np.full_like(low, np.nan, dtype=np.float64)
    
    for i in range(window-1, len(high)):
        upper[i] = np.max(high[i-window+1:i+1])
        lower[i] = np.min(low[i-window+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = np.full_like(close_12h, np.nan, dtype=np.float64)
    
    # Calculate EMA with proper smoothing
    alpha = 2.0 / (50 + 1)
    ema_50_12h[49] = np.mean(close_12h[:50])  # First EMA value
    for i in range(50, len(close_12h)):
        ema_50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_50_12h[i-1]
    
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels on 4h (20-period)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # Volume filter: volume > 1.8x average
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        trend = ema_50_12h_aligned[i]
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above upper band + trend up + volume spike
            if price_now > upper_band and price_now > trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below lower band + trend down + volume spike
            elif price_now < lower_band and price_now < trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below upper band or trend turns down
            if price_now < upper_band or price_now < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above lower band or trend turns up
            if price_now > lower_band or price_now > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0
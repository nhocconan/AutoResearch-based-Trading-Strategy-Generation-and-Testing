#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with Weekly Trend and Volume Spike.
Long when price breaks above 20-bar high + weekly trend up + volume spike.
Short when price breaks below 20-bar low + weekly trend down + volume spike.
Exit when price crosses opposite Donchian boundary (20-bar low for long, high for short).
Weekly trend defined as price above/below 50-period EMA on weekly timeframe.
Designed for low frequency (12-37 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.empty_like(close_1w, dtype=np.float64)
    ema_1w.fill(np.nan)
    # Calculate EMA properly with alpha
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_1w[i] = close_1w[i]
        elif np.isnan(ema_1w[i-1]):
            ema_1w[i] = close_1w[i]
        else:
            ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align weekly EMA to 6h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: volume > 2.0x average (to avoid false breakouts)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), weekly EMA (50), volume MA (20)
    start_idx = max(19, 50, 19) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        upper = donch_high[i]
        lower = donch_low[i]
        trend_1w = ema_1w_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above upper Donchian + weekly trend up + volume spike
            if price_now > upper and price_now > trend_1w and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below lower Donchian + weekly trend down + volume spike
            elif price_now < lower and price_now < trend_1w and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian (opposite boundary)
            if price_now < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above upper Donchian (opposite boundary)
            if price_now > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_20_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0
EOF
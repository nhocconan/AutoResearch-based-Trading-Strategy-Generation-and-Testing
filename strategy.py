#!/usr/bin/env python3
"""
6h Donchian Breakout with 1d Volume Spike and 1w Trend Filter.
Long when: 1) Price breaks above Donchian(20) high, 2) 1d volume > 2x average, 3) Price > 1w EMA100.
Short when: 1) Price breaks below Donchian(20) low, 2) 1d volume > 2x average, 3) Price < 1w EMA100.
Exit when: 1) Price crosses Donchian midpoint (mean reversion) OR 2) 1w EMA100 trend reverses.
Designed for 6h timeframe: targets 50-150 total trades over 4 years (12-37/year).
Works in bull (breakouts) and bear (mean reversion on trend reversal).
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
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    donch_mid = (donch_high + donch_low) / 2
    
    # 1d volume for spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-day MA
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1w EMA100 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_100_1w = np.full(len(close_1w), np.nan)
    for i in range(99, len(close_1w)):
        ema_100_1w[i] = np.mean(close_1w[i-99:i+1])  # Simple MA for speed, min_periods=100
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian(20), 1d vol MA, 1w EMA
    start_idx = max(19, 19, 99)  # Donchian(20), 1d vol MA(20), 1w EMA(100)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_100_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        vol_1d_ma = vol_ma_1d_aligned[i]
        vol_1d_now = volume[i]  # Current 6h volume, compare to 1d average
        ema_trend = ema_100_1w_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        mid = donch_mid[i]
        
        # Volume filter: current 6h volume > 2x 1d average volume
        # Note: This approximates volume spike; 1d vol is daily total, 6h is 1/4 of day
        vol_filter = vol_1d_now > 0.5 * vol_1d_ma  # Adjusted for timeframe difference
        
        if position == 0:
            # Long: breakout above Donchian high + volume spike + above 1w EMA100
            if price > upper and vol_filter and price > ema_trend:
                signals[i] = size
                position = 1
            # Short: breakdown below Donchian low + volume spike + below 1w EMA100
            elif price < lower and vol_filter and price < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses Donchian midpoint OR trend turns bearish
            if price < mid or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses Donchian midpoint OR trend turns bullish
            if price > mid or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_Breakout_1dVolumeSpike_1wEMA100"
timeframe = "6h"
leverage = 1.0
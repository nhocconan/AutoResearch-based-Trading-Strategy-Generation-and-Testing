#!/usr/bin/env python3
"""
12h Weekly Donchian Breakout with Volume Spike and Trend Filter.
Long when: 1) Price breaks above weekly Donchian high (20-period), 2) Price > 12h EMA50 (bullish trend), 3) Volume > 2x 20-period average.
Short when: 1) Price breaks below weekly Donchian low (20-period), 2) Price < 12h EMA50 (bearish trend), 3) Volume > 2x 20-period average.
Exit when price returns to weekly midline or trend reverses.
Designed for 12h timeframe: targets 50-150 total trades over 4 years (12-37/year).
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
    
    # Get weekly data for Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_20_1w = np.full(len(high_1w), np.nan, dtype=np.float64)
    low_20_1w = np.full(len(low_1w), np.nan, dtype=np.float64)
    for i in range(19, len(high_1w)):
        high_20_1w[i] = np.max(high_1w[i-19:i+1])
        low_20_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Align weekly Donchian levels to 12h timeframe
    high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, high_20_1w)
    low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, low_20_1w)
    
    # Weekly midline for exit (average of high and low)
    weekly_midline = (high_20_1w + low_20_1w) / 2.0
    weekly_midline_aligned = align_htf_to_ltf(prices, df_1w, weekly_midline)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly Donchian (20 periods), 12h EMA (50 periods), volume MA (20 periods)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20_1w_aligned[i]) or np.isnan(low_20_1w_aligned[i]) or 
            np.isnan(weekly_midline_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        donchian_high = high_20_1w_aligned[i]
        donchian_low = low_20_1w_aligned[i]
        midline = weekly_midline_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + bullish trend + volume spike
            if price > donchian_high and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly Donchian low + bearish trend + volume spike
            elif price < donchian_low and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly midline or trend turns bearish
            if price <= midline or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to weekly midline or trend turns bullish
            if price >= midline or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Weekly_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0
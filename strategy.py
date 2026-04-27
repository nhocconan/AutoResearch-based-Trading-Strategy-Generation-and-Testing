#!/usr/bin/env python3
"""
12h Donchian Breakout with 1-week Trend Filter and Volume Confirmation.
Long when price breaks above Donchian(20) high + weekly trend up + volume spike.
Short when price breaks below Donchian(20) low + weekly trend down + volume spike.
Exit when price reverses to middle of Donchian channel or trend changes.
Designed for low frequency (12-37 trades/year) to minimize fee drag.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend
    weekly_close = df_1w['close'].values
    ema_50_1w = np.empty_like(weekly_close, dtype=np.float64)
    ema_50_1w.fill(np.nan)
    if len(weekly_close) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(weekly_close[:50])
        for i in range(50, len(weekly_close)):
            ema_50_1w[i] = alpha * weekly_close[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channel (20-period)
    donchian_period = 20
    donchian_high = np.empty_like(high, dtype=np.float64)
    donchian_low = np.empty_like(high, dtype=np.float64)
    donchian_high.fill(np.nan)
    donchian_low.fill(np.nan)
    for i in range(donchian_period - 1, n):
        donchian_high[i] = np.max(high[i-donchian_period+1:i+1])
        donchian_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Donchian middle (exit signal)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume filter: volume > 1.5x average (calculated from 12h volume MA30)
    vol_ma_30 = np.empty_like(volume, dtype=np.float64)
    vol_ma_30.fill(np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20 periods) and weekly EMA (50 periods)
    start_idx = max(donchian_period - 1, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        upper = donchian_high[i]
        lower = donchian_low[i]
        middle = donchian_mid[i]
        weekly_trend = ema_50_1w_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_30[i]
        
        # Breakout conditions
        breakout_up = price_now > upper
        breakout_down = price_now < lower
        
        if position == 0:
            # Bull: breakout above upper + weekly trend up + volume
            if breakout_up and price_now > weekly_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: breakout below lower + weekly trend down + volume
            elif breakout_down and price_now < weekly_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle or weekly trend turns down
            if price_now < middle or price_now < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle or weekly trend turns up
            if price_now > middle or price_now > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_20_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0
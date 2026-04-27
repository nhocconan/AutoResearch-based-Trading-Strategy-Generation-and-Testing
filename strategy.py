#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ehlers Fisher Transform with weekly trend filter and volume confirmation.
Trades Fisher reversals when weekly trend confirms direction and volume exceeds 1-week average.
Designed to capture reversals in both bull and bear markets by using weekly trend as filter.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Ehlers Fisher Transform on 6h close prices
    price = close
    # Normalize price to 0-1 range over lookback period
    length = 10
    highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
    lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
    
    # Avoid division by zero
    diff = highest_high - lowest_low
    diff[diff == 0] = 1e-10
    
    # Normalize price
    value1 = 0.33 * 2 * ((price - lowest_low) / diff - 0.5)
    # Smooth the value
    value2 = pd.Series(value1).ewm(alpha=0.5, adjust=False).mean().values
    
    # Fisher Transform
    # Clamp to prevent extreme values
    value2 = np.clip(value2, -0.999, 0.999)
    fish = 0.5 * np.log((1 + value2) / (1 - value2))
    # Smooth Fisher
    fish = pd.Series(fish).ewm(alpha=0.5, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Fisher, weekly EMA, and daily volume MA
    start_idx = max(length, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(fish[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current Fisher value and previous
        fish_now = fish[i]
        fish_prev = fish[i-1]
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_50_1w_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Fisher reversal with volume and weekly trend alignment
        if position == 0:
            # Long: Fisher crosses above -1.5 with volume + weekly uptrend
            if fish_now > -1.5 and fish_prev <= -1.5 and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: Fisher crosses below +1.5 with volume + weekly downtrend
            elif fish_now < 1.5 and fish_prev >= 1.5 and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Fisher crosses above +1.5 or weekly trend turns down
            if fish_now >= 1.5 or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Fisher crosses below -1.5 or weekly trend turns up
            if fish_now <= -1.5 or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_FisherTransform_1wTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ehlers Fisher Transform with daily volume confirmation and weekly trend filter.
Enters long when Fisher crosses above -1.5 with above-average volume and weekly uptrend.
Enters short when Fisher crosses below +1.5 with above-average volume and weekly downtrend.
Fisher Transform identifies extreme price movements and reversals, effective in ranging markets.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def fisher_transform(price, period=10):
    """Ehlers Fisher Transform"""
    if len(price) < period:
        return np.full_like(price, np.nan, dtype=float)
    
    # Normalize price to [-1, 1] range
    highest = np.max(price)
    lowest = np.min(price)
    if highest == lowest:
        return np.zeros_like(price)
    
    value = 2 * ((price - lowest) / (highest - lowest) - 0.5)
    # Clamp to [-0.999, 0.999] to avoid infinities
    value = np.clip(value, -0.999, 0.999)
    
    # Fisher transform
    fish = 0.5 * np.log((1 + value) / (1 - value))
    
    # Smoothed with delay
    if len(fish) < 2:
        return fish
    
    smoothed = np.full_like(fish, np.nan)
    smoothed[0] = fish[0]
    for i in range(1, len(fish)):
        smoothed[i] = 0.5 * fish[i] + 0.5 * smoothed[i-1]
    
    return smoothed

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Typical price for Fisher
    typical_price = (high + low + close) / 3.0
    
    # Calculate 6h Fisher Transform
    fish = fisher_transform(typical_price, 10)
    fish_sma = pd.Series(fish).rolling(window=5, min_periods=5).mean().values
    
    # Align Fisher to 6h
    fish_aligned = align_htf_to_ltf(prices, prices, fish_sma)
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Fisher, volume MA, and weekly EMA
    start_idx = max(10, 5, 20, 21)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(fish_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 6h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_21_1w_aligned[i]
        
        # Current Fisher value
        fish_now = fish_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Fisher signals
        fish_cross_above = (i > start_idx and 
                           fish_aligned[i-1] <= -1.5 and fish_now > -1.5)
        fish_cross_below = (i > start_idx and 
                           fish_aligned[i-1] >= 1.5 and fish_now < 1.5)
        
        # Entry conditions
        if position == 0:
            # Long: Fisher crosses above -1.5 with volume + weekly uptrend
            if fish_cross_above and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: Fisher crosses below +1.5 with volume + weekly downtrend
            elif fish_cross_below and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Fisher crosses below +1.5 or weekly trend turns down
            if fish_now < 1.5 or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Fisher crosses above -1.5 or weekly trend turns up
            if fish_now > -1.5 or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_FisherTransform_1dVolume_1wTrend"
timeframe = "6h"
leverage = 1.0
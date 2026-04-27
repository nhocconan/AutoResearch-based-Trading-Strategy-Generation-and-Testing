#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian(20) breakout with 1-week volume confirmation and 1-week EMA trend filter.
Trades breakouts when price breaks above/below 20-day high/low with volume above weekly average
and weekly trend alignment. Designed to work in both bull and bear markets by using weekly trend
as filter and volume to confirm breakout strength. Target: 15-25 trades/year per symbol (60-100 total)
to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volume filter and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly volume MA(20)
    vol_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate weekly EMA(25) for trend
    close_1w = df_1w['close'].values
    ema_25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_25_1w)
    
    # Calculate daily Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian channels, weekly volume MA, and weekly EMA
    start_idx = max(20, 20, 25)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i]) or np.isnan(ema_25_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1w_aligned[i]
        trend_1w = ema_25_1w_aligned[i]
        
        # Current Donchian levels
        upper = high_20[i]
        lower = low_20[i]
        
        # Volume filter: volume > 1.3x weekly average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and weekly trend alignment
        if position == 0:
            # Long: price breaks above upper with volume + weekly uptrend
            if price_now > upper and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price breaks below lower with volume + weekly downtrend
            elif price_now < lower and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midline or weekly trend turns down
            midline = (upper + lower) / 2
            if price_now < midline or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midline or weekly trend turns up
            midline = (upper + lower) / 2
            if price_now > midline or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wVolume_1wTrend"
timeframe = "1d"
leverage = 1.0
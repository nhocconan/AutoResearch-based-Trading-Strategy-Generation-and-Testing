#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day volume confirmation and 1-week trend filter.
Trades breakouts above/below the 12-hour Donchian(20) when volume exceeds the 1-day average and the weekly trend aligns.
Designed to work in both bull and bear markets by using weekly trend as filter and volume to confirm breakout strength.
Target: 12-37 trades/year per symbol (50-150 total over 4 years) to minimize fee drag.
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
    
    # Get 12-hour data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour Donchian(20) channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian, volume MA, and weekly EMA
    start_idx = max(20, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 12-hour price and volume
        high_12h_now = high_12h[i // 2] if i // 2 < len(high_12h) else high_12h[-1]
        low_12h_now = low_12h[i // 2] if i // 2 < len(low_12h) else low_12h[-1]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_50_1w_aligned[i]
        
        # Donchian breakout conditions
        upper_break = high_12h_now > donchian_high_aligned[i]
        lower_break = low_12h_now < donchian_low_aligned[i]
        
        # Volume filter: volume > 1.5x 1-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and weekly trend alignment
        if position == 0:
            # Long: breakout above upper Donchian + volume + weekly uptrend
            if upper_break and vol_filter and close[i] > trend_1w:
                signals[i] = size
                position = 1
            # Short: breakout below lower Donchian + volume + weekly downtrend
            elif lower_break and vol_filter and close[i] < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midline or weekly trend turns down
            midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] < midline or close[i] < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian midline or weekly trend turns up
            midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] > midline or close[i] > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Volume_1dTrendFilter_1wTrend"
timeframe = "12h"
leverage = 1.0
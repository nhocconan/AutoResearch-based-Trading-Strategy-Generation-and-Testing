#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day volume confirmation and 1-week trend filter.
Trades breakouts above the 20-period Donchian high with volume above 1-day average and weekly uptrend,
or breakdowns below the 20-period Donchian low with volume above 1-day average and weekly downtrend.
Designed to work in both bull and bear markets by using weekly trend as filter and volume to confirm breakout strength.
Target: 12-37 trades/year per symbol (48-148 total over 4 years) to minimize fee drift.
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
    
    # Get 12-hour data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already aligned via get_htf_data)
    donchian_high_12h = donchian_high
    donchian_low_12h = donchian_low
    
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
    
    # Calculate 1-week EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian channels, volume MA, and weekly EMA
    start_idx = max(20, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_50_1w_aligned[i]
        
        # Current Donchian levels
        donchian_high_now = donchian_high_12h[i]
        donchian_low_now = donchian_low_12h[i]
        
        # Volume filter: volume > 1.3x 1-day average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and weekly trend alignment
        if position == 0:
            # Long: price breaks above Donchian high with volume + weekly uptrend
            if price_now > donchian_high_now and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume + weekly downtrend
            elif price_now < donchian_low_now and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low or weekly trend turns down
            if price_now < donchian_low_now or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high or weekly trend turns up
            if price_now > donchian_high_now or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_DonchianBreakout_1dVolume_1wTrend"
timeframe = "12h"
leverage = 1.0
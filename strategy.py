#!/usr/bin/env python3
"""
1d Donchian breakout with weekly EMA trend filter and volume confirmation.
Trades breakouts of daily Donchian(20) channels when price is above/below
weekly EMA(20) and volume exceeds 1.5x daily average.
Designed to work in both bull and bear markets via trend filter.
Target: 10-25 trades/year per symbol (40-100 total over 4 years).
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
    
    # Get daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian, weekly EMA, and volume MA
    start_idx = max(20, 20, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
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
        vol_ma = vol_ma_20_1d_aligned[i]
        weekly_trend = ema_20_1w_aligned[i]
        
        # Current Donchian levels
        upper = dh_aligned[i]
        lower = dl_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and weekly trend alignment
        if position == 0:
            # Long: price breaks above upper band with volume + weekly uptrend
            if price_now > upper and vol_filter and price_now > weekly_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band with volume + weekly downtrend
            elif price_now < lower and vol_filter and price_now < weekly_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midline or weekly trend turns down
            midline = (upper + lower) / 2
            if price_now < midline or price_now < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midline or weekly trend turns up
            midline = (upper + lower) / 2
            if price_now > midline or price_now > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0
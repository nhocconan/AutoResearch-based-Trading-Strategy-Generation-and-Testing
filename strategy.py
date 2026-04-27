#!/usr/bin/env python3
"""
12h Donchian(20) breakout with 1w trend filter and volume confirmation.
Enters long when price breaks above 1w Donchian upper with above-average volume.
Enters short when price breaks below 1w Donchian lower with above-average volume.
Uses 1w trend to filter direction, reducing whipsaw in range/chop markets.
Designed for low trade frequency (<30/year) to minimize fee drag on 12h timeframe.
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
    
    # Get weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume MA(20)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly Donchian, volume MA, and weekly EMA
    start_idx = max(20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        trend_1w = ema_50_1w_aligned[i]
        
        # Current weekly Donchian levels
        donch_high_now = donch_high_aligned[i]
        donch_low_now = donch_low_aligned[i]
        
        # Volume filter: volume > 1.5x 12h average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Weekly Donchian breakout with volume and weekly trend alignment
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume + weekly uptrend
            if price_now > donch_high_now and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly Donchian low with volume + weekly downtrend
            elif price_now < donch_low_now and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly Donchian mid or weekly trend turns down
            donch_mid = (donch_high_now + donch_low_now) / 2.0
            if price_now <= donch_mid or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to weekly Donchian mid or weekly trend turns up
            donch_mid = (donch_high_now + donch_low_now) / 2.0
            if price_now >= donch_mid or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0
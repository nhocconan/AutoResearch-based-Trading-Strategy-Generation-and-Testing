#!/usr/bin/env python3
"""
1-day Donchian(20) breakout with 1-week trend filter and volume confirmation.
Enters long when price breaks above 20-day high with volume above average and 1-week uptrend.
Enters short when price breaks below 20-day low with volume above average and 1-week downtrend.
Uses 1-day timeframe for entries, 1-week for trend filter.
Target: 10-20 trades/year per symbol to minimize fee drag and work in both bull and bear markets.
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
    
    # Get daily data (same as primary timeframe since we're using 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels
    high_20d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly close for trend filter
    close_1w = df_1w['close'].values
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20d[i]) or np.isnan(low_20d[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20[i]
        trend_1w = close_1w_aligned[i]
        
        # Volume filter: volume > 1.5x 20-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions
        if position == 0:
            # Long: break above 20-day high with volume + 1-week uptrend
            if price_now > high_20d[i] and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: break below 20-day low with volume + 1-week downtrend
            elif price_now < low_20d[i] and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below 20-day low or 1-week trend turns down
            if price_now < low_20d[i] or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above 20-day high or 1-week trend turns up
            if price_now > high_20d[i] or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0
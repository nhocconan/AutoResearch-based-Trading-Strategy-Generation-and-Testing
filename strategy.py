#!/usr/bin/env python3
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
    
    # Get 1d data for Donchian channel and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channel
    highest_20d = np.full(len(high_1d), np.nan)
    lowest_20d = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        highest_20d[i] = np.max(high_1d[i-19:i+1])
        lowest_20d[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 20-day average volume
    avg_volume_20d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        avg_volume_20d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 4-hour EMA50 for trend filter
    ema_period = 50
    ema_4h = np.full(n, np.nan)
    if n >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema_4h[i] = (close[i] * (2 / (ema_period + 1)) + 
                         ema_4h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align Donchian levels and average volume to 4h timeframe
    highest_20d_aligned = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20d)
    avg_volume_20d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, volume average, and EMA
    start_idx = max(19, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20d_aligned[i]) or np.isnan(lowest_20d_aligned[i]) or 
            np.isnan(avg_volume_20d_aligned[i]) or np.isnan(ema_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_20d_aligned[i]
        upper_band = highest_20d_aligned[i]
        lower_band = lowest_20d_aligned[i]
        ema_trend = ema_4h[i]
        
        if position == 0:
            # Long: Break above upper Donchian band with volume confirmation and uptrend
            if (price > upper_band and vol > 1.5 * avg_vol and price > ema_trend):
                signals[i] = size
                position = 1
            # Short: Break below lower Donchian band with volume confirmation and downtrend
            elif (price < lower_band and vol > 1.5 * avg_vol and price < ema_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price re-enters Donchian channel or trend fails
            if price < upper_band or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price re-enters Donchian channel or trend fails
            if price > lower_band or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Trend_Volume
Breakout strategy using Donchian channels on 12h timeframe.
Long when price breaks above Donchian upper band (20-period) and price > 1d EMA50 (uptrend) and volume > 1.5x average volume.
Short when price breaks below Donchian lower band (20-period) and price < 1d EMA50 (downtrend) and volume > 1.5x average volume.
Exit when price crosses back to Donchian middle band or trend filter fails.
Uses 1d trend filter (EMA50) to avoid counter-trend trades in strong trends.
Target: 15-35 trades/year per symbol.
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
    
    # Donchian channel parameters
    donchian_period = 20
    
    # Calculate Donchian upper and lower bands
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(donchian_period - 1, n):
        highest_high[i] = np.max(high[i - donchian_period + 1:i + 1])
        lowest_low[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Calculate average volume for volume filter
    vol_avg_period = 20
    vol_avg = np.full(n, np.nan)
    if n >= vol_avg_period:
        vol_avg[vol_avg_period - 1] = np.mean(volume[:vol_avg_period])
        for i in range(vol_avg_period, n):
            vol_avg[i] = (volume[i] * (2 / (vol_avg_period + 1)) + 
                         vol_avg[i - 1] * (1 - (2 / (vol_avg_period + 1))))
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_1d_period + 1))))
    
    # Align 1d EMA50 to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian bands, volume average, and EMA1d
    start_idx = max(donchian_period - 1, vol_avg_period - 1, ema_1d_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        middle_band = (upper_band + lower_band) / 2
        vol_avg_val = vol_avg[i]
        ema1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band, uptrend, and high volume
            if (price > upper_band and price > ema1d_val and vol > 1.5 * vol_avg_val):
                signals[i] = size
                position = 1
            # Short: price breaks below lower band, downtrend, and high volume
            elif (price < lower_band and price < ema1d_val and vol > 1.5 * vol_avg_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below middle band or trend fails
            if price < middle_band or price < ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above middle band or trend fails
            if price > middle_band or price > ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0
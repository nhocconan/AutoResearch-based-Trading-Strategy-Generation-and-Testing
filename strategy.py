#!/usr/bin/env python3
# 6h_12h_donchian_volume_filter_v1
# Hypothesis: 6-hour Donchian(20) breakout with 12-hour volume confirmation and 1-day EMA50 trend filter.
# Long when price breaks above 20-period high with volume > 1.5x 20-bar average and price > daily EMA50.
# Short when price breaks below 20-period low with volume > 1.5x 20-bar average and price < daily EMA50.
# Exit when price returns to the opposite Donchian level (20-period low for longs, 20-period high for shorts).
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extreme levels.
# Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_donchian_volume_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume moving average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = np.full(len(vol_12h), np.nan)
    vol_sum = 0
    for i in range(len(vol_12h)):
        vol_sum += vol_12h[i]
        if i >= 20:
            vol_sum -= vol_12h[i-20]
        if i >= 19:
            vol_ma_20_12h[i] = vol_sum / 20
    
    # Align 12h volume MA to 6h timeframe
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema = close_1d[49]  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50_1d[49] = ema
        for i in range(50, len(close_1d)):
            ema = (close_1d[i] - ema) * multiplier + ema
            ema_50_1d[i] = ema
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    
    # Calculate rolling max/min for high/low
    for i in range(n):
        if i >= 19:  # Need 20 periods for Donchian(20)
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below 20-period low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above 20-period high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume and trend filters
            if (close[i] > donchian_high[i] and 
                volume[i] > vol_ma_20_12h_aligned[i] * 1.5 and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume and trend filters
            elif (close[i] < donchian_low[i] and 
                  volume[i] > vol_ma_20_12h_aligned[i] * 1.5 and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
12h_1d_Donchian_Breakout_Volume_Trend_v1
Hypothesis: On 12h timeframe, use Donchian(20) breakout for trend following with volume confirmation and 1d EMA trend filter.
Buy when price breaks above 20-period high with above-average volume and price > 1d EMA50.
Sell when price breaks below 20-period low with above-average volume and price < 1d EMA50.
This captures strong momentum moves while filtering counter-trend noise. Designed for 12h timeframe to reduce trade frequency
and minimize fee drag, targeting 15-30 trades/year. Works in both bull and bear markets by following the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 50-period EMA on 1d data
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])  # SMA seed
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_50_1d[i-1] * (49 / (50 + 1)))
    
    # Load 12h data for Donchian channels and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period Donchian channels on 12h data
    highest_20 = np.full_like(high_12h, np.nan)
    lowest_20 = np.full_like(low_12h, np.nan)
    for i in range(19, len(high_12h)):
        highest_20[i] = np.max(high_12h[i-19:i+1])
        lowest_20[i] = np.min(low_12h[i-19:i+1])
    
    # Calculate 20-period average volume on 12h data
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(19, len(volume_12h)):
        vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align indicators to 12h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_20)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 50)  # Donchian needs 20, EMA needs 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_ratio = volume_12h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for long entries: price breaks above Donchian high with volume confirmation and uptrend
            if (close[i] > highest_20_aligned[i] and 
                volume_ratio > 1.5 and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Look for short entries: price breaks below Donchian low with volume confirmation and downtrend
            elif (close[i] < lowest_20_aligned[i] and 
                  volume_ratio > 1.5 and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low (trend reversal)
            if close[i] < lowest_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high (trend reversal)
            if close[i] > highest_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0
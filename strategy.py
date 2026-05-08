#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# Donchian channels identify breakout points with clear support/resistance levels.
# 12h EMA > 50-period provides trend direction filter to avoid counter-trend trades.
# Volume confirmation ensures institutional participation in breakouts.
# This combination works in both bull and bear markets by trading breakouts in the direction of the trend.
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.

name = "4h_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    lookback = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest[i] = np.max(high[i-lookback:i])
        lowest[i] = np.min(low[i-lookback:i])
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # EMA(50) on 12h close
    ema_50 = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50[i] = (close_12h[i] * 2/51) + (ema_50[i-1] * 49/51)
    
    # Volume average on 12h (20-period for ~10 days)
    vol_ma_12h = np.full_like(volume_12h, np.nan)
    if len(volume_12h) >= 20:
        vol_ma_12h[19] = np.mean(volume_12h[:20])
        for i in range(20, len(volume_12h)):
            vol_ma_12h[i] = (volume_12h[i] * 2/21) + (vol_ma_12h[i-1] * 19/21)
    
    # Align 12h indicators to 4h timeframe
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50)
    vol_ma_12h_4h = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma_12h_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, above EMA, volume above average
            if close[i] > highest[i] and close[i] > ema_50_4h[i] and volume[i] > vol_ma_12h_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, below EMA, volume above average
            elif close[i] < lowest[i] and close[i] < ema_50_4h[i] and volume[i] > vol_ma_12h_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or below EMA
            if close[i] < lowest[i] or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high or above EMA
            if close[i] > highest[i] or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Uses 12h EMA50 for trend alignment, 6h Donchian(20) for breakout entry, and volume spike (>1.7x 20-bar avg) for confirmation.
# Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 12h trend direction and requiring volume confirmation to avoid false breakouts.

name = "6h_Donchian20_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels from prior period (primary TF)
    lookback = 1
    # Donchian upper/lower based on prior 20 periods high/low
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(lookback).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(lookback).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper, close > 12h EMA50, volume spike (>1.7x avg)
            if (high[i] > highest_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.7 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower, close < 12h EMA50, volume spike (>1.7x avg)
            elif (low[i] < lowest_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.7 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Donchian lower or volume drops
            if (low[i] < lowest_low[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Donchian upper or volume drops
            if (high[i] > highest_high[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRVolumeRegime
Hypothesis: 4h Donchian(20) breakout with ATR-based regime filter and volume confirmation. 
Enters long when price breaks above upper Donchian band in low-volatility regime (ATR ratio < 0.8) with volume spike.
Enters short when price breaks below lower Donchian band in low-volatility regime with volume spike.
Exits when price reverts to middle Donchian band (20-period mean).
Uses 12h timeframe for trend filter (price > EMA50 for longs, price < EMA50 for shorts) to avoid counter-trend trades.
Designed for 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
Uses discrete position sizing (0.25) to minimize churn. Works in both bull and bear markets by following 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle = (upper + lower) / 2.0
    
    # Calculate ATR (14-period) for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR mean) for regime filter
    atr_mean = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_mean
    
    # Load 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period Donchian + 50-period ATR mean + 50-period EMA)
    start_idx = max(20, 50, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above upper Donchian + low volatility regime + volume spike + bullish 12h trend
        if (close[i] > upper[i] and atr_ratio[i] < 0.8 and volume_spike[i] and 
            close[i] > ema_50_12h_aligned[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below lower Donchian + low volatility regime + volume spike + bearish 12h trend
        elif (close[i] < lower[i] and atr_ratio[i] < 0.8 and volume_spike[i] and 
              close[i] < ema_50_12h_aligned[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to middle Donchian band
        elif position == 1 and close[i] < middle[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > middle[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_ATRVolumeRegime"
timeframe = "4h"
leverage = 1.0
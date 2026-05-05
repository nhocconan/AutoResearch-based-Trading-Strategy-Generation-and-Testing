#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (1.5x)
# Long when price breaks above Donchian upper AND price > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower AND price < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price reverts to Donchian midpoint
# Uses 4h timeframe with 1d HTF for robust trend filtering (target: 75-200 total over 4 years)
# Donchian provides clear price channel structure from 20-period highs/lows
# Volume confirmation reduces false breakouts
# 1d EMA50 offers strong trend filter effective in both bull and bear markets
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure

name = "4h_Donchian20_1dEMA50_VolumeSpike_1.5x"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    if len(high) >= 20:
        upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        midpoint = (upper + lower) / 2.0
    else:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        midpoint = np.full(n, np.nan)
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation on 4h (threshold: 1.5x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(midpoint[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper AND price > EMA50 AND volume spike
            if (close[i] > upper[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower AND price < EMA50 AND volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to midpoint
            if close[i] < midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to midpoint
            if close[i] > midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
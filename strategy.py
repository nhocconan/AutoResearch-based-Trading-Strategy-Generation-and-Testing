#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation (1.5x)
# Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 1.5x 20-day average
# Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 1.5x 20-day average
# Exit when price reverts to 10-day EMA OR 1w EMA50 filter reverses
# Uses Donchian channels for clear breakout structure + volume confirmation to reduce false signals
# 1w EMA50 provides higher timeframe trend filter effective in both bull and bear markets
# Designed for 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Timeframe: 1d (primary), HTF: 1w

name = "1d_Donchian20_1wEMA50_VolumeSpike_1.5x"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on 1d
    if len(high) >= 20:
        highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        highest_20 = np.full(n, np.nan)
        lowest_20 = np.full(n, np.nan)
    
    # Calculate 10-period EMA for exit
    if len(close) >= 10:
        ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    else:
        ema_10 = np.full(n, np.nan)
    
    # Calculate 1w EMA(50)
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1w = np.full(len(close_1w), np.nan)
    
    # Align HTF indicators to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation on 1d (threshold: 1.5x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_10[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high AND price > 1w EMA50 AND volume spike
            if (close[i] > highest_20[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND price < 1w EMA50 AND volume spike
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to 10-day EMA OR price < 1w EMA50 (trend weakening)
            if close[i] < ema_10[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to 10-day EMA OR price > 1w EMA50 (trend weakening)
            if close[i] > ema_10[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
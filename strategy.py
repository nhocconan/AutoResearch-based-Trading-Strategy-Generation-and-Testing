#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and volume confirmation (1.5x)
# Long when price breaks above 20-period high AND price > 4h EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below 20-period low AND price < 4h EMA50 AND volume > 1.5x 20-period average
# Exit when price reverts to 20-period midpoint OR 4h EMA50 filter reverses
# Uses Donchian channels for clear breakout structure + volume confirmation to reduce false signals
# 4h EMA50 provides higher timeframe trend filter effective in both bull and bear markets
# Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Timeframe: 1h (primary), HTF: 4h

name = "1h_Donchian20_4hEMA50_VolumeSpike_1.5x"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(50)
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation on 1h (threshold: 1.5x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Calculate Donchian channels on 1h (20-period high/low)
    if len(high) >= 20:
        high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (high_roll + low_roll) / 2.0
    else:
        high_roll = np.full(n, np.nan)
        low_roll = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high AND price > EMA50 AND volume spike
            if (close[i] > high_roll[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 20-period low AND price < EMA50 AND volume spike
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price reverts to midpoint OR price < EMA50 (trend weakening)
            if close[i] < donchian_mid[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price reverts to midpoint OR price > EMA50 (trend weakening)
            if close[i] > donchian_mid[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
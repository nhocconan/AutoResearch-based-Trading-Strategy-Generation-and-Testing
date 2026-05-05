#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA50 trend filter and volume confirmation (1.5x)
# Long when price breaks above 12h Donchian upper (20) AND price > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below 12h Donchian lower (20) AND price < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price reverts to 12h Donchian midpoint OR 1d EMA50 filter reverses
# Uses Donchian channels for clear breakout structure + volume confirmation to reduce false signals
# 1d EMA50 provides higher timeframe trend filter effective in both bull and bear markets
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 12h (primary), HTF: 1d

name = "12h_Donchian20_1dEMA50_VolumeSpike_1.5x"
timeframe = "12h"
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
    
    # Calculate 12h Donchian channels (20-period)
    if len(high) >= 20:
        high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = high_roll_max
        donchian_lower = low_roll_min
        donchian_mid = (donchian_upper + donchian_lower) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation on 12h (threshold: 1.5x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > EMA50 AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < EMA50 AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to Donchian midpoint OR price < EMA50 (trend weakening)
            if close[i] < donchian_mid[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to Donchian midpoint OR price > EMA50 (trend weakening)
            if close[i] > donchian_mid[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
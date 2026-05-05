#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w EMA50 trend filter and volume confirmation (2.0x)
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 2.0x 20-period average
# Exit when price reverts to Donchian(20) midpoint OR 1w EMA50 filter reverses
# Uses Donchian channels for clear structure + volume confirmation to reduce false signals
# 1w EMA50 provides higher timeframe trend filter effective in both bull and bear markets
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 12h (primary), HTF: 1w

name = "12h_Donchian20_1wEMA50_VolumeSpike_2.0x"
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
    
    # Get 1w data ONCE before loop for Donchian and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) from 1w data
    # Upper = max(high_1w over 20 periods), Lower = min(low_1w over 20 periods)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, mid_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation on 12h (threshold: 2.0x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > EMA50 AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < EMA50 AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to midpoint OR price < EMA50 (trend weakening)
            if close[i] < donchian_mid_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to midpoint OR price > EMA50 (trend weakening)
            if close[i] > donchian_mid_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
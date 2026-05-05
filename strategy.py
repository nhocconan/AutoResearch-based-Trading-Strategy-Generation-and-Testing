#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation
# Long when: price breaks above Donchian upper band (20-period high) AND 1d EMA50 rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian lower band (20-period low) AND 1d EMA50 falling AND volume > 1.5x 20-period MA
# Exit when: price crosses Donchian middle band (10-period average of upper/lower) OR volume drops below average
# Uses Donchian for breakout structure, 1d EMA50 for trend filter, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 12h
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_rising = np.zeros(len(ema_50), dtype=bool)
        ema_50_falling = np.zeros(len(ema_50), dtype=bool)
        for i in range(1, len(ema_50)):
            if not np.isnan(ema_50[i]) and not np.isnan(ema_50[i-1]):
                ema_50_rising[i] = ema_50[i] > ema_50[i-1]
                ema_50_falling[i] = ema_50[i] < ema_50[i-1]
    else:
        ema_50 = np.full(len(close_1d), np.nan)
        ema_50_rising = np.zeros(len(close_1d), dtype=bool)
        ema_50_falling = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d EMA50 trend to 12h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising.astype(float))
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling.astype(float))
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
        volume_normal = volume > (0.5 * vol_ma_20)  # for exit condition
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_normal = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(volume_normal[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper band + rising EMA50 + volume filter
            if (close[i] > donchian_upper[i] and 
                ema_50_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + falling EMA50 + volume filter
            elif (close[i] < donchian_lower[i] and 
                  ema_50_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: cross below middle band OR volume drops below normal
            if (close[i] < donchian_middle[i] or volume_normal[i] == 0.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: cross above middle band OR volume drops below normal
            if (close[i] > donchian_middle[i] or volume_normal[i] == 0.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
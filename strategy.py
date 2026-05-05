#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume confirmation
# Long when: close > Donchian upper(20) AND 12h EMA50 rising AND volume > 1.5x 20-period MA
# Short when: close < Donchian lower(20) AND 12h EMA50 falling AND volume > 1.5x 20-period MA
# Exit when: price crosses Donchian middle (10-period average) OR volume drops below average
# Uses Donchian for breakouts, 12h EMA for HTF trend filter, volume for conviction
# Timeframe: 4h, HTF: 12h. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian_12hEMA_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 4h
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Get 12h data ONCE before loop for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h
    close_12h = df_12h['close'].values
    if len(close_12h) >= 50:
        ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
        # Determine trend: rising if current > previous, falling if current < previous
        ema_rising = np.zeros(len(ema_50), dtype=bool)
        ema_falling = np.zeros(len(ema_50), dtype=bool)
        for i in range(1, len(ema_50)):
            if not np.isnan(ema_50[i]) and not np.isnan(ema_50[i-1]):
                ema_rising[i] = ema_50[i] > ema_50[i-1]
                ema_falling[i] = ema_50[i] < ema_50[i-1]
    else:
        ema_50 = np.full(len(close_12h), np.nan)
        ema_rising = np.zeros(len(close_12h), dtype=bool)
        ema_falling = np.zeros(len(close_12h), dtype=bool)
    
    # Align 12h EMA trend to 4h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling.astype(float))
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema_rising_aligned[i]) or 
            np.isnan(ema_falling_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above upper + rising 12h EMA + volume filter
            if (close[i] > donchian_upper[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below lower + falling 12h EMA + volume filter
            elif (close[i] < donchian_lower[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle OR volume drops below average
            if (close[i] < donchian_middle[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle OR volume drops below average
            if (close[i] > donchian_middle[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
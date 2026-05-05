#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w EMA50 trend filter + volume spike confirmation
# Long when: price breaks above Donchian(20) upper band AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) lower band AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-period MA
# Exit when: price returns to Donchian(20) middle band OR trend reverses
# Uses Donchian for structure, 1w EMA for trend filter, volume spike for conviction
# Timeframe: 12h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_Breakout_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 12h (20-period)
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = highest_high
        donchian_lower = lowest_low
        donchian_middle = (highest_high + lowest_low) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Calculate volume confirmation on 12h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on 1w timeframe
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_rising = np.diff(ema_50_1w, prepend=np.nan) > 0
        ema_falling = np.diff(ema_50_1w, prepend=np.nan) < 0
    else:
        ema_rising = np.full(len(close_1w), False)
        ema_falling = np.full(len(close_1w), False)
    
    # Align 1w EMA trend to 12h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + 1w EMA rising + volume filter
            if (close[i] > donchian_upper[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + 1w EMA falling + volume filter
            elif (close[i] < donchian_lower[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian middle OR 1w EMA turns falling
            if (close[i] <= donchian_middle[i] or ema_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian middle OR 1w EMA turns rising
            if (close[i] >= donchian_middle[i] or ema_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
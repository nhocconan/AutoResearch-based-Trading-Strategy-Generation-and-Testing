#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA34 trend filter + volume confirmation.
Long when price breaks above Donchian upper band AND 12h EMA34 rising AND volume > 1.3x average.
Short when price breaks below Donchian lower band AND 12h EMA34 falling AND volume > 1.3x average.
Exit when price reverts to Donchian middle (20-period mean).
Uses 4h for Donchian calculation and 12h for EMA trend filter to reduce whipsaw and capture multi-timeframe alignment.
Target: 75-200 total trades over 4 years (19-50/year). Donchian breakouts capture trends,
12h EMA filter ensures higher-timeframe trend alignment, volume confirmation filters fakeouts.
Works in bull markets (captures uptrends with 12h EMA up) and bear markets (captures downtrends with 12h EMA down).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels on 4h timeframe (20-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = ((donchian_upper + donchian_lower) / 2).values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_prev = np.roll(ema_12h, 1)
    ema_12h_prev[0] = ema_12h[0]
    ema_12h_rising = ema_12h > ema_12h_prev  # True when EMA is rising
    ema_12h_falling = ema_12h < ema_12h_prev  # True when EMA is falling
    
    # Align 4h Donchian to 4h timeframe (no alignment needed)
    donchian_upper_aligned = donchian_upper
    donchian_lower_aligned = donchian_lower
    donchian_middle_aligned = donchian_middle
    
    # Align 12h EMA trend to 4h timeframe
    ema_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_rising)
    ema_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_falling)
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_12h_rising_aligned[i]) or 
            np.isnan(ema_12h_falling_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        du = donchian_upper_aligned[i]
        dl = donchian_lower_aligned[i]
        dm = donchian_middle_aligned[i]
        ema_up = ema_12h_rising_aligned[i]
        ema_down = ema_12h_falling_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND 12h EMA34 rising AND volume > 1.3x avg
            if price > du and ema_up and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND 12h EMA34 falling AND volume > 1.3x avg
            elif price < dl and ema_down and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle
            if price < dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian middle
            if price > dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0
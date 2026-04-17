#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above Donchian upper (20) AND 12h EMA(34) rising AND 4h volume > 1.5x 20-bar average.
Short when price breaks below Donchian lower (20) AND 12h EMA(34) falling AND 4h volume > 1.5x 20-bar average.
Exit when price touches Donchian middle (20-bar midpoint) or opposite band.
Uses 12h for trend filter (HTF) and 4h for execution, volume, and Donchian bands.
Designed to capture strong trends with volume confirmation in both bull and bear markets.
Target: 20-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34)
    ema_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_12h_rising = np.diff(ema_12h, prepend=ema_12h[0]) > 0
    ema_12h_falling = np.diff(ema_12h, prepend=ema_12h[0]) < 0
    
    # Align 12h EMA trend to 4h
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_falling)
    
    # Calculate 4h Donchian(20) bands
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    middle = (upper + lower) / 2
    
    # Calculate 4h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper[i]
        breakout_lower = close[i] < lower[i]
        
        # Exit conditions: touch middle band or opposite band
        touch_middle = abs(close[i] - middle[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < lower[i]) or \
                         (position == -1 and close[i] > upper[i])
        
        if position == 0:
            # Long: break above upper with volume confirmation and rising 12h EMA
            if (breakout_upper and volume_confirmed and ema_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower with volume confirmation and falling 12h EMA
            elif (breakout_lower and volume_confirmed and ema_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch middle or break below lower
            if (touch_middle or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch middle or break above upper
            if (touch_middle or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_Volume_Trend"
timeframe = "4h"
leverage = 1.0
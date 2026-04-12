#!/usr/bin/env python3
"""
1d_1w_donchian_breakout_volume_v1
Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
Long when price breaks above upper band AND weekly EMA21 is rising AND volume > 1.5x average.
Short when price breaks below lower band AND weekly EMA21 is falling AND volume > 1.5x average.
Exit when price crosses opposite Donchian band or volume dries up.
Designed for low trade frequency (<15/year) to minimize fee drag in both bull and bear markets.
"""

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA21 for trend direction
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_prev = np.roll(ema_21_1w, 1)
    ema_21_1w_prev[0] = np.nan
    weekly_rising = ema_21_1w > ema_21_1w_prev
    weekly_falling = ema_21_1w < ema_21_1w_prev
    
    # Align weekly trend to daily
    weekly_rising_aligned = align_htf_to_ltf(prices, df_1w, weekly_rising)
    weekly_falling_aligned = align_htf_to_ltf(prices, df_1w, weekly_falling)
    
    # Daily Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(weekly_rising_aligned[i]) or np.isnan(weekly_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: break above upper band + weekly uptrend + volume
        if (close[i] > highest_20[i] and weekly_rising_aligned[i] and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: break below lower band + weekly downtrend + volume
        elif (close[i] < lowest_20[i] and weekly_falling_aligned[i] and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit long: price breaks below lower band
        elif position == 1 and close[i] < lowest_20[i]:
            position = 0
            signals[i] = 0.0
        # Exit short: price breaks above upper band
        elif position == -1 and close[i] > highest_20[i]:
            position = 0
            signals[i] = 0.0
        # Optional exit: volume dries up (< 0.5x average)
        elif position == 1 and volume[i] < (vol_ma[i] * 0.5):
            position = 0
            signals[i] = 0.0
        elif position == -1 and volume[i] < (vol_ma[i] * 0.5):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
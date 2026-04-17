#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
Long when price breaks above upper Donchian channel with volume > 1.3x 12h avg volume AND 1d EMA50 rising.
Short when price breaks below lower Donchian channel with volume > 1.3x 12h avg volume AND 1d EMA50 falling.
Exit when price touches the opposite Donchian channel or reverses against trend.
Uses 12h for execution and volume, 1d for EMA trend filter and Donchian calculation.
Designed to capture medium-term trends with volume confirmation, working in both bull and bear markets.
Target: 12-30 trades/year per symbol.
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
    
    # Get 1d data for EMA trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_rising = ema_50_1d > np.roll(ema_50_1d, 1)
    ema_50_falling = ema_50_1d < np.roll(ema_50_1d, 1)
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    
    # Calculate Donchian channels (20-period) on 1d data
    # Upper channel: highest high over last 20 periods
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over last 20 periods
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    # Calculate 12h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(upper_20_aligned[i]) or
            np.isnan(lower_20_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-bar average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper_20_aligned[i]
        breakout_lower = close[i] < lower_20_aligned[i]
        
        # Exit conditions
        exit_long = (close[i] < lower_20_aligned[i]) or (position == 1 and not ema_50_rising_aligned[i])
        exit_short = (close[i] > upper_20_aligned[i]) or (position == -1 and not ema_50_falling_aligned[i])
        
        if position == 0:
            # Long: break above upper Donchian with volume confirmation and rising 1d EMA50
            if breakout_upper and volume_confirmed and ema_50_rising_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume confirmation and falling 1d EMA50
            elif breakout_lower and volume_confirmed and ema_50_falling_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian or EMA50 turns falling
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian or EMA50 turns rising
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0
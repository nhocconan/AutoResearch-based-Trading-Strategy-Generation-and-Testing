#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume spike
# Long when price breaks above upper Donchian channel with 1d ADX > 25 and volume > 2x average
# Short when price breaks below lower Donchian channel with 1d ADX > 25 and volume > 2x average
# Exit when price returns to the middle of the Donchian channel or reverses to opposite side
# Uses Donchian for breakout signals, ADX for trend strength, volume for conviction
# Designed to capture strong trending moves while avoiding choppy markets
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_Donchian_Breakout_1dADX_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels
    upper_channel = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Align Donchian channels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle_channel)
    
    # Calculate 1d ADX for trend strength filter
    # Calculate True Range
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(values, period):
        smoothed = np.full_like(values, np.nan, dtype=float)
        if len(values) >= period:
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr = wilder_smoothing(tr, 20)
    plus_di = 100 * wilder_smoothing(plus_dm, 20) / atr
    minus_di = 100 * wilder_smoothing(minus_dm, 20) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smoothing(dx, 20)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, ADX > 25, volume spike
            if (close[i] > upper_aligned[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, ADX > 25, volume spike
            elif (close[i] < lower_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle or breaks below lower channel
            if (close[i] <= middle_aligned[i]) or (close[i] < lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle or breaks above upper channel
            if (close[i] >= middle_aligned[i]) or (close[i] > upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
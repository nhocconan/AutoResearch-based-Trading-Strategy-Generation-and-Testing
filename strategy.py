#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above 12h upper Donchian band (20-period) AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Short when price breaks below 12h lower Donchian band (20-period) AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Exit when price crosses back below/above the opposite Donchian band (upper for long exit, lower for short exit).
# Uses 12h timeframe to reduce trade frequency and avoid fee drag. ADX ensures we only trade in strong trends.
# Volume confirmation filters out false breakouts. Designed for 15-30 trades/year to minimize costs.
# Works in both bull and bear markets by following the trend direction via ADX.

name = "12h_Donchian_20_ADX25_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 14-period ADX calculation on daily timeframe
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    tr_14 = pd.Series(tr.values).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm.values).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm.values).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 12h timeframe (wait for daily bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Sufficient warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, ADX > 25, volume filter
            long_cond = (close[i] > high_20[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: price breaks below lower Donchian, ADX > 25, volume filter
            short_cond = (close[i] < low_20[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below lower Donchian band
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above upper Donchian band
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
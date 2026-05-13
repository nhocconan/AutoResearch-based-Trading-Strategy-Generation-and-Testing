#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_Confirmation
Hypothesis: Use daily Donchian channel (20-day high/low) breakouts with volume confirmation and ADX trend filter. 
Go long when price breaks above 20-day high with above-average volume and ADX > 25 (trending).
Go short when price breaks below 20-day low with above-average volume and ADX > 25.
Exit when price crosses the 20-day midpoint or ADX falls below 20 (range).
Designed for 12h timeframe to limit trades and avoid fee drag while capturing multi-day trends.
Works in both bull (breakout longs) and bear (breakout shorts) markets.
"""

name = "12h_Donchian20_Breakout_Volume_Confirmation"
timeframe = "12h"
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
    
    # Get daily data for Donchian channel and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channel
    high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Calculate ADX for trend strength
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr.values
    minus_di = 100 * minus_dm_smooth / atr.values
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume average (20-period) and align
    vol_avg_20 = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with volume confirmation and trending market
            if (close[i] > donchian_high_aligned[i] and 
                volume[i] > vol_avg_aligned[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with volume confirmation and trending market
            elif (close[i] < donchian_low_aligned[i] and 
                  volume[i] > vol_avg_aligned[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below midpoint or trend weakens
            if close[i] < donchian_mid_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above midpoint or trend weakens
            if close[i] > donchian_mid_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
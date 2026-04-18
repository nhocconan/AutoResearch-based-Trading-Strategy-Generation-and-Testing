#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + Daily ADX Filter
Strategy: Breakout of 20-period Donchian channel with volume confirmation and daily trend filter.
Long when price breaks above upper band with volume spike and daily ADX > 25.
Short when price breaks below lower band with volume spike and daily ADX > 25.
Designed for low trade frequency with strong breakout edge in both bull and bear markets.
"""

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
    
    # Donchian channel (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Get daily data for ADX filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values (Wilder's smoothing)
        def WilderMA(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) >= period:
                result[period-1] = np.nansum(arr[:period])
                for i in range(period, len(arr)):
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        tr_smooth = WilderMA(tr, period)
        plus_dm_smooth = WilderMA(plus_dm, period)
        minus_dm_smooth = WilderMA(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
        minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = WilderMA(dx, period)
        return adx
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with volume spike and daily uptrend
            if (close[i] > donchian_high[i] and volume_spike[i] and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume spike and daily uptrend
            elif (close[i] < donchian_low[i] and volume_spike[i] and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Hold long position
            signals[i] = 0.25
            # Exit: price breaks below middle of channel or volatility drop
            if close[i] < (donchian_high[i] + donchian_low[i]) / 2:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Hold short position
            signals[i] = -0.25
            # Exit: price breaks above middle of channel
            if close[i] > (donchian_high[i] + donchian_low[i]) / 2:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_VolumeSpike_DailyADX"
timeframe = "12h"
leverage = 1.0
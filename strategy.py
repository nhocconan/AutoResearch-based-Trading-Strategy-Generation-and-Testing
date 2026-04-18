#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + ADX Trend Filter
Breaks out of Donchian channels with volume confirmation and ADX trend filter.
Designed for low trade frequency with strong trend-following edge.
Works in both bull and bear markets by capturing strong trending moves.
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
    
    # Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_channel, lower_channel = donchian_channels(high, low, 20)
    
    # ADX calculation for trend strength
    def calculate_adx(high, low, close, period=14):
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
        
        # Smoothed values using Wilder's smoothing (EMA-like)
        def wilder_smoothing(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) >= period:
                result[period-1] = np.nansum(arr[:period])
                for i in range(period, len(arr)):
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        tr_smooth = wilder_smoothing(tr, period)
        plus_dm_smooth = wilder_smoothing(plus_dm, period)
        minus_dm_smooth = wilder_smoothing(minus_dm, period)
        
        plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
        minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilder_smoothing(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume spike (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # need enough history for Donchian and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long breakout: price breaks above upper channel with volume and trend
            if price > upper_channel[i] and volume_spike[i] and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower channel with volume and trend
            elif price < lower_channel[i] and volume_spike[i] and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold until breakdown
            signals[i] = 0.25
            # Exit: price breaks below lower channel (reversal signal)
            if price < lower_channel[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until breakout
            signals[i] = -0.25
            # Exit: price breaks above upper channel (reversal signal)
            if price > upper_channel[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0
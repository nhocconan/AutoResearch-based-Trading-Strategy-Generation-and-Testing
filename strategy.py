#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and ADX Filter
Hypothesis: Donchian channels identify breakout points where price breaks recent highs/lows.
Combined with volume spikes (institutional participation) and ADX > 25 (trending market),
this captures strong momentum moves. The strategy works in both bull and bear markets
by going long on upside breaks and short on downside breaks. Low trade frequency due
to the confluence of three conditions reduces fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels (upper and lower bands)"""
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(len(high)):
        if i >= period - 1:
            upper[i] = np.max(high[i - period + 1:i + 1])
            lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder smoothing
    def Wilder_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        # Initial value: simple average
        result[period - 1] = np.mean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period
        return result
    
    atr = Wilder_smoothing(tr, period)
    plus_di = 100 * Wilder_smoothing(plus_dm, period) / np.where(atr != 0, atr, 1)
    minus_di = 100 * Wilder_smoothing(minus_dm, period) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = Wilder_smoothing(dx, period)
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    upper, lower = calculate_donchian_channels(high, low, 20)
    
    # Calculate ADX for trend strength
    adx = calculate_adx(high, low, close, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i - 19):i + 1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i - 19:i + 1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators (20 for Donchian + extra for ADX)
    
    for i in range(start_idx, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band + ADX > 25 + volume spike
            if (close[i] > upper[i] and 
                adx[i] > 25 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian band + ADX > 25 + volume spike
            elif (close[i] < lower[i] and 
                  adx[i] > 25 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below the midpoint of the channel or ADX weakens
            midpoint = (upper[i] + lower[i]) / 2
            if close[i] < midpoint or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above the midpoint of the channel or ADX weakens
            midpoint = (upper[i] + lower[i]) / 2
            if close[i] > midpoint or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0
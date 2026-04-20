#!/usr/bin/env python3
"""
4h_ADX_Strength_Trend_Breakout
Hypothesis: Trade only when ADX > 25 (strong trend) and price breaks Donchian(20) with volume > 1.8x average.
Long on upper breakout, short on lower breakout. Exit when ADX < 20 or opposite breakout.
Uses ADX to filter weak/choppy markets and volume to confirm breakout strength.
Designed for 4h timeframe to keep trades ~30-60/year, avoiding fee overload.
Works in bull/bear: ADX ensures we only trade strong trends, reducing whipsaws.
"""

name = "4h_ADX_Strength_Trend_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        # Smoothed values
        def smooth_rma(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) >= period:
                result[period-1] = np.mean(arr[:period])
                for i in range(period, len(arr)):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_smooth = smooth_rma(tr, period)
        plus_dm_smooth = smooth_rma(plus_dm, period)
        minus_dm_smooth = smooth_rma(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx[:] = np.where(tr_smooth != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = smooth_rma(dx, period)
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    upper = rolling_max(high, 20)
    lower = rolling_min(low, 20)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_strong = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure ADX and channels are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: ADX > 25 (strong trend), price breaks above upper band, strong volume
            if adx[i] > 25 and close[i] > upper[i] and volume_strong[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 (strong trend), price breaks below lower band, strong volume
            elif adx[i] > 25 and close[i] < lower[i] and volume_strong[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: ADX < 20 (weakening trend) OR price breaks below lower band
            if adx[i] < 20 or close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: ADX < 20 (weakening trend) OR price breaks above upper band
            if adx[i] < 20 or close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
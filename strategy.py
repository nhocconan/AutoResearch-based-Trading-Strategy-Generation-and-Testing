#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter.
# Long when price breaks above upper Donchian channel with volume > 1.5x 4h average volume and ADX > 25.
# Short when price breaks below lower Donchian channel with volume > 1.5x 4h average volume and ADX > 25.
# Exit when price crosses back below the middle of the Donchian channel (20-period average of high/low).
# Uses Donchian for breakout signals, volume for confirmation, ADX for trend strength.
# Target: 20-50 trades/year per symbol to stay within frequency limits.
name = "4h_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    upper_channel = rolling_max(high, 20)
    lower_channel = rolling_min(low, 20)
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        def smooth(arr, window):
            res = np.full_like(arr, np.nan, dtype=float)
            for i in range(len(arr)):
                if i < window:
                    if i == 0:
                        res[i] = arr[i]
                    else:
                        res[i] = np.nansum(arr[:i+1]) / (i+1)
                else:
                    res[i] = res[i-1] - (res[i-1] / window) + (arr[i] / window)
            return res
        
        atr = smooth(tr, window)
        dm_plus_smooth = smooth(dm_plus, window)
        dm_minus_smooth = smooth(dm_minus, window)
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / atr
        minus_di = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx[np.isnan(plus_di) | np.isnan(minus_di) | (plus_di + minus_di) == 0] = 0
        
        adx = smooth(dx, window)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full_like(volume, np.nan, dtype=float)
    for i in range(len(volume)):
        if i < 20:
            if i == 0:
                vol_ma_20[i] = volume[i]
            else:
                vol_ma_20[i] = np.mean(volume[:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure Donchian and ADX are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(middle_channel[i]) or np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above upper Donchian with volume and ADX > 25
            if price > upper_channel[i] and volume_confirmed and adx[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian with volume and ADX > 25
            elif price < lower_channel[i] and volume_confirmed and adx[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below middle Donchian channel
            if price < middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above middle Donchian channel
            if price > middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals